# SQLi-only binary pipeline (Benign vs SQLi) on Kaggle Modified_SQL_Dataset.csv
# Protocol follows SQLi_XSS_Comparative_Study.ipynb: normalisation, dedup before split, char 1-3-gram TF-IDF,
# tuning on a 10,000-row stratified subsample (3-fold CV, macro-F1), 80:20/70:30/90:10 splits, feature ablation,
# repeated stratified 5x2 CV with Friedman + Holm-corrected Wilcoxon, feature importance, external sqlmap test.
import os, sys, re, html, json, time, hashlib, platform
import numpy as np, pandas as pd
from urllib.parse import unquote_plus
from scipy import stats
import sklearn, xgboost
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve)
DATA = sys.argv[1] if len(sys.argv) > 1 else 'data/Modified_SQL_Dataset.csv'
SQLMAP = sys.argv[2] if len(sys.argv) > 2 else 'sqlmap_payloads.csv'
OUT = sys.argv[3] if len(sys.argv) > 3 else 'results_sqli'
os.makedirs(OUT, exist_ok=True)
RS = 42; N_JOBS = -1; TUNE_SUB = 10000
SPLITS = {'80:20': 0.2, '70:30': 0.3, '90:10': 0.1}; MAIN = '80:20'
def log(*a): print(*a, flush=True)
info = dict(python=platform.python_version(), sklearn=sklearn.__version__, xgboost=xgboost.__version__,
            numpy=np.__version__, pandas=pd.__version__, scipy=__import__('scipy').__version__, cpus=os.cpu_count())
# ---------------- data ----------------
sha = hashlib.sha256(open(DATA, 'rb').read()).hexdigest()
head = open(DATA, 'rb').read(4)
enc = 'utf-16' if head[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig'
raw = pd.read_csv(DATA, encoding=enc)                       # no skipped rows
txtcol = [c for c in raw.columns if c.lower() in ('query', 'sentence')][0]
raw = raw.rename(columns={txtcol: 'text', 'Label': 'label'})[['text', 'label']]
info.update(file=os.path.basename(DATA), sha256=sha, encoding=enc, rows=len(raw), columns=list(raw.columns),
            raw_labels={str(k): int(v) for k, v in raw.label.value_counts(dropna=False).items()})
log('data', info)
def normalize(s):
    s = str(s)
    try: s = unquote_plus(s)
    except Exception: pass
    s = html.unescape(s).lower()
    return re.sub(r'\s+', ' ', s).strip()
clog = [('Raw data', len(raw))]
df = raw.dropna(subset=['text']).copy(); clog.append(('After dropping missing payloads', len(df)))
df['text'] = df.text.astype(str); df = df[df.text.str.strip() != '']
df['norm'] = df.text.map(normalize); df = df[df.norm != '']; clog.append(('After dropping empty payloads', len(df)))
df['label'] = df.label.astype(int)
conf = df.groupby('norm').label.nunique(); ct = set(conf[conf > 1].index)
nconf_rows = int(df.norm.isin(ct).sum())
df = df[~df.norm.isin(ct)]; clog.append((f'After removing {len(ct)} label-conflicting texts ({nconf_rows} rows)', len(df)))
df = df.drop_duplicates('norm').reset_index(drop=True); clog.append(('After removing exact duplicates (normalised)', len(df)))
pd.DataFrame(clog, columns=['Step', 'Rows']).to_csv(f'{OUT}/cleaning_log.csv', index=False)
df[['text', 'norm', 'label']].to_csv(f'{OUT}/preprocessed_dataset.csv', index=False)
X = df.norm.values; y = df.label.values
log(clog, np.bincount(y))
# lexical profile (cleaned data)
SQLK = r'\b(select|union|insert|update|delete|drop|from|where|or|and|sleep|benchmark|waitfor|exec|having|order|group|null|like)\b'
lex = pd.DataFrame({'label': y,
    'length': [len(t) for t in X],
    'sql_keywords': [len(re.findall(SQLK, t)) for t in X],
    'single_quotes': [t.count("'") for t in X],
    'parentheses': [t.count('(') + t.count(')') for t in X],
    'comment_marks': [len(re.findall(r'--|#|/\*', t)) for t in X],
    'spaced_equals': [t.count(' = ') for t in X],
    'equals': [t.count('=') for t in X],
    'select_from': [bool(re.search(r'\bselect\b.*\bfrom\b', t)) for t in X],
    'starts_like_fragment': [bool(re.match(r"^[\d'\"\)\(\-]", t)) for t in X]})
lex.groupby('label').mean().to_csv(f'{OUT}/lexical_profile.csv')
# ---------------- splits ----------------
SPLIT = {}; rows = []
for n, ts in SPLITS.items():
    tr, te = train_test_split(np.arange(len(y)), test_size=ts, stratify=y, random_state=RS)
    assert not (set(X[tr]) & set(X[te])); SPLIT[n] = (tr, te)
    rows.append(dict(Split=n, Train=len(tr), Test=len(te), **{'Train Benign': int((y[tr] == 0).sum()), 'Train SQLi': int((y[tr] == 1).sum()),
                'Test Benign': int((y[te] == 0).sum()), 'Test SQLi': int((y[te] == 1).sum())}))
pd.DataFrame(rows).to_csv(f'{OUT}/split_sizes.csv', index=False)
# ---------------- models ----------------
def vec(kind='char'):
    w = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b|[^\w\s]', ngram_range=(1, 2), max_features=20000, sublinear_tf=True, lowercase=False, dtype=np.float32)
    c = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=20000, sublinear_tf=True, lowercase=False, dtype=np.float32)
    return {'word': w, 'char': c, 'word+char': FeatureUnion([('word', w), ('char', c)])}[kind]
MODELS = {
 'KNN': dict(est=lambda: KNeighborsClassifier(algorithm='brute', n_jobs=N_JOBS),
             space={'clf__n_neighbors': [3, 5, 7, 9], 'clf__weights': ['uniform', 'distance'], 'clf__metric': ['cosine', 'euclidean']}, search='grid'),
 'SVM': dict(est=lambda: SVC(cache_size=1000, random_state=RS),
             space={'clf__kernel': ['linear', 'rbf'], 'clf__C': [0.1, 1, 10], 'clf__gamma': ['scale']}, search='grid'),
 'Random Forest': dict(est=lambda: RandomForestClassifier(random_state=RS, n_jobs=N_JOBS),
             space={'clf__n_estimators': [100, 200, 300], 'clf__max_depth': [None, 50, 100], 'clf__min_samples_split': [2, 5],
                    'clf__min_samples_leaf': [1, 2], 'clf__class_weight': [None, 'balanced']}, search='random', n_iter=8),
 'XGBoost': dict(est=lambda: XGBClassifier(objective='binary:logistic', eval_metric='logloss', tree_method='hist', random_state=RS, n_jobs=N_JOBS, verbosity=0),
             space={'clf__n_estimators': [100, 200], 'clf__max_depth': [4, 6, 8], 'clf__learning_rate': [0.1, 0.3],
                    'clf__subsample': [0.8, 1.0], 'clf__colsample_bytree': [0.5, 0.8]}, search='random', n_iter=6, sw=True)}
NAMES = list(MODELS)
def pipe_of(n, params=None, feats='char'):
    p = Pipeline([('vec', vec(feats)), ('clf', MODELS[n]['est']())])
    if params: p.set_params(**params)
    return p
def fkw(n, yy): return {'clf__sample_weight': compute_sample_weight('balanced', yy)} if MODELS[n].get('sw') else {}
def score_of(p, Xs):
    if isinstance(p.named_steps['clf'], SVC): return p.decision_function(Xs)
    return p.predict_proba(Xs)[:, 1]
# ---------------- tuning ----------------
BEST = {}; tr_rows = []
for sn, (tr, _) in SPLIT.items():
    sub, _ = train_test_split(tr, train_size=TUNE_SUB, stratify=y[tr], random_state=RS)
    cv = StratifiedKFold(3, shuffle=True, random_state=RS)
    for n in NAMES:
        m = MODELS[n]; t0 = time.time()
        if m['search'] == 'grid': s = GridSearchCV(pipe_of(n), m['space'], scoring='f1_macro', cv=cv, n_jobs=1, refit=False)
        else: s = RandomizedSearchCV(pipe_of(n), m['space'], n_iter=m['n_iter'], scoring='f1_macro', cv=cv, n_jobs=1, refit=False, random_state=RS)
        s.fit(X[sub], y[sub], **fkw(n, y[sub])); BEST[(sn, n)] = s.best_params_
        tr_rows.append(dict(Split=sn, Algorithm=n, cv_macro_f1=s.best_score_, params=json.dumps({k[5:]: v for k, v in s.best_params_.items()}), seconds=time.time() - t0))
        log('tune', tr_rows[-1])
pd.DataFrame(tr_rows).to_csv(f'{OUT}/best_hyperparameters.csv', index=False)
# ---------------- evaluation ----------------
res = []; ROC = {}; FITTED = {}
for sn, (tr, te) in SPLIT.items():
    for n in NAMES:
        p = pipe_of(n, BEST[(sn, n)]); t0 = time.time(); p.fit(X[tr], y[tr], **fkw(n, y[tr])); tt = time.time() - t0
        t0 = time.time(); pr = p.predict(X[te]); tp_ = time.time() - t0
        sc = score_of(p, X[te]); tn, fp, fn, tp = confusion_matrix(y[te], pr, labels=[0, 1]).ravel()
        res.append(dict(Algorithm=n, Split=sn, Accuracy=accuracy_score(y[te], pr), Precision=precision_score(y[te], pr), Recall=recall_score(y[te], pr),
                        F1=f1_score(y[te], pr), F1_macro=f1_score(y[te], pr, average='macro'), ROC_AUC=roc_auc_score(y[te], sc),
                        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp), FPR=fp / (fp + tn), FNR=fn / (fn + tp),
                        train_s=tt, predict_s=tp_, predict_ms_per_payload=tp_ * 1000 / len(te)))
        log('eval', res[-1])
        if sn == MAIN:
            FITTED[n] = p; ROC[n] = sc
            if n == 'SVM': pd.DataFrame({'y': y[te], 'norm': X[te], **{}}).to_csv(f'{OUT}/test_80_20.csv', index=False)
pd.DataFrame(res).to_csv(f'{OUT}/results_all_splits.csv', index=False)
tr, te = SPLIT[MAIN]
pd.DataFrame({'y': y[te], **{n: ROC[n] for n in NAMES}}).to_csv(f'{OUT}/roc_scores_80_20.csv', index=False)
pd.DataFrame({'norm': X[te], 'y': y[te], **{n: FITTED[n].predict(X[te]) for n in NAMES}}).to_csv(f'{OUT}/test_predictions_80_20.csv', index=False)
# feature importance
fi = []
for n in ['Random Forest', 'XGBoost']:
    p = FITTED[n]; names = p.named_steps['vec'].get_feature_names_out(); imp = p.named_steps['clf'].feature_importances_
    for r, i in enumerate(np.argsort(imp)[::-1][:20]): fi.append(dict(Algorithm=n, rank=r + 1, ngram=names[i], importance=float(imp[i])))
pd.DataFrame(fi).to_csv(f'{OUT}/feature_importance_top20.csv', index=False)
# ---------------- ablation ----------------
ab = []
for feats in ['word', 'char', 'word+char']:
    for n in NAMES:
        p = pipe_of(n, BEST[(MAIN, n)], feats); t0 = time.time(); p.fit(X[tr], y[tr], **fkw(n, y[tr])); tt = time.time() - t0
        pr = p.predict(X[te]); ab.append(dict(Features=feats, Algorithm=n, F1=f1_score(y[te], pr), F1_macro=f1_score(y[te], pr, average='macro'), train_s=tt))
        log('abl', ab[-1])
pd.DataFrame(ab).to_csv(f'{OUT}/ablation_features.csv', index=False)
# ---------------- repeated CV + tests ----------------
cvr = []
for fold, (a, b) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RS).split(X, y)):
    for n in NAMES:
        p = pipe_of(n, BEST[(MAIN, n)]); p.fit(X[a], y[a], **fkw(n, y[a])); pr = p.predict(X[b])
        cvr.append(dict(Fold=fold, Algorithm=n, Accuracy=accuracy_score(y[b], pr), F1=f1_score(y[b], pr), F1_macro=f1_score(y[b], pr, average='macro')))
    log('cv fold', fold, [round(r['F1_macro'], 4) for r in cvr[-4:]])
cvd = pd.DataFrame(cvr); cvd.to_csv(f'{OUT}/repeated_cv_folds.csv', index=False)
cvd.groupby('Algorithm')[['Accuracy', 'F1', 'F1_macro']].agg(['mean', 'std']).reindex(NAMES).to_csv(f'{OUT}/repeated_cv_summary.csv')
F = cvd.pivot(index='Fold', columns='Algorithm', values='F1_macro')[NAMES]; nf, k = F.shape
chi, pf = stats.friedmanchisquare(*[F[c].values for c in NAMES]); W = chi / (nf * (k - 1))
ranks = F.rank(axis=1, ascending=False).mean().reindex(NAMES)
pairs = []
for i in range(k):
    for j in range(i + 1, k):
        A, B = NAMES[i], NAMES[j]; d = F[A].values - F[B].values; nz = d[d != 0]
        if len(nz) == 0: w, p, rb = 0.0, 1.0, 0.0
        else:
            w, p = stats.wilcoxon(F[A].values, F[B].values, zero_method='wilcox', alternative='two-sided')
            rk = stats.rankdata(np.abs(nz)); rb = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()
        pairs.append(dict(Comparison=f'{A} vs {B}', mean_dF1=d.mean(), W=w, p=p, rank_biserial=rb))
pw = pd.DataFrame(pairs).sort_values('p').reset_index(drop=True); run = 0; adj = []
for i, p in enumerate(pw.p): run = max(run, min(1.0, (len(pw) - i) * p)); adj.append(run)
pw['p_holm'] = adj; pw['significant'] = pw.p_holm < 0.05; pw.to_csv(f'{OUT}/wilcoxon_pairwise.csv', index=False)
json.dump(dict(friedman_chi2=chi, friedman_p=pf, kendall_w=W, n_folds=int(nf), mean_ranks=ranks.to_dict(),
               first_place_counts=F.rank(axis=1, ascending=False).eq(1).sum().reindex(NAMES).to_dict()), open(f'{OUT}/friedman_test.json', 'w'), indent=2, default=float)
log('friedman', chi, pf, W, ranks.to_dict()); log(pw)
# ---------------- external sqlmap test ----------------
if os.path.exists(SQLMAP):
    S = pd.read_csv(SQLMAP); S['norm'] = S.payload.map(normalize); n_gen = len(S)
    S = S.drop_duplicates('norm'); n_uni = len(S); known = set(X); S = S[~S.norm.isin(known)]
    def spaced(t):
        t = re.sub(r'\s*([=(),<>])\s*', r' \1 ', t); return re.sub(r'\s+', ' ', t).strip()
    S['spaced'] = S.norm.map(spaced)
    for n in NAMES: S[n] = FITTED[n].predict(S.norm.values); S[n + ' (spaced)'] = FITTED[n].predict(S.spaced.values)
    S.to_csv(f'{OUT}/external_sqlmap_predictions.csv', index=False)
    ext = dict(generated=n_gen, unique=n_uni, in_dataset=n_uni - len(S), evaluated=len(S), level1=int((S.level == 1).sum()),
               detection_all={n: float(S[n].mean() * 100) for n in NAMES},
               detection_spaced={n: float(S[n + ' (spaced)'].mean() * 100) for n in NAMES},
               share_spaced_equals={'orig': float(S.norm.str.contains(' = ').mean() * 100), 'spaced': float(S.spaced.str.contains(' = ').mean() * 100)},
               detection_level1={n: float(S[S.level == 1][n].mean() * 100) for n in NAMES},
               by_technique={t: {n: float(g[n].mean() * 100) for n in NAMES} | {'n': len(g)} for t, g in S.groupby('technique')})
    json.dump(ext, open(f'{OUT}/external_sqlmap_summary.json', 'w'), indent=2); log('external', ext)
json.dump(info, open(f'{OUT}/run_info.json', 'w'), indent=2)
log('DONE')
