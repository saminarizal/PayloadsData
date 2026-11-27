# -*- coding: utf-8 -*-
"""
Tambahan eksperimen REV10 untuk naskah JISI (deteksi SQLi, dataset Kaggle sajid576).

  Butir 5  Baseline pembanding
           - Logistic Regression dan Multinomial Naive Bayes (TF-IDF char 1-3, split 80:20 yang sama)
           - libinjection (libinjection-python) dan OWASP CRS v4.30.0 rule file 942 (SQLi),
             dievaluasi dengan anomaly scoring CRS (CRITICAL = 5, WARNING = 3, ambang 5) pada PL1 dan PL2
  Butir 6  Ablasi normalisasi spasi operator: normalize_sql_operators() diterapkan pada data latih,
           test set Kaggle, dan payload sqlmap; semua model dilatih ulang dengan hyperparameter yang sama.

Protokol mengikuti sqli_pipeline.py (REV9): normalisasi dasar (URL decode, HTML unescape, huruf kecil,
rapatkan spasi), buang teks kosong & berlabel ganda, deduplikasi sebelum split, split bertingkat 80:20
seed 42, TF-IDF char 1-3 (20.000 fitur, sublinear, dilatih hanya pada data latih).

Pemakaian (Colab):
  !pip -q install libinjection-python==1.1.6 regex tabulate
  !python sqli_baselines_norm.py --data "$DATA_PATH" --sqlmap sqlmap_payloads.csv \
        --hp results_sqli/best_hyperparameters.csv --ref results_sqli --out results_rev10
Keluaran: CSV per tabel, tables_rev10.md (Tabel A dan B), run_info_rev10.json.
"""
import os, re, sys, io, json, time, html, argparse, hashlib, platform, urllib.request
import numpy as np, pandas as pd
from urllib.parse import unquote_plus
from scipy.stats import binomtest
import sklearn, xgboost
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

RS, TUNE_SUB, TEST_SIZE = 42, 10000, 0.20
CRS_TAG = 'v4.30.0'
CRS_URL = f'https://raw.githubusercontent.com/coreruleset/coreruleset/{CRS_TAG}/rules/REQUEST-942-APPLICATION-ATTACK-SQLI.conf'
def log(*a): print(*a, flush=True)

# =====================================================================================
# 1. Normalisasi
# =====================================================================================
def normalize(s):
    """Normalisasi dasar REV9 (dipakai semua kondisi)."""
    s = str(s)
    try: s = unquote_plus(s)
    except Exception: pass
    s = html.unescape(s).lower()
    return re.sub(r'\s+', ' ', s).strip()

# Token multi-karakter didahulukan agar '--' tidak terpecah menjadi '-' '-', '/*' menjadi '/' '*', dst.
_OPS = r"!=|<>|<=|>=|--|/\*|\*/|[=<>(),'\"+\-*/]"
_OP_RE = re.compile(r'\s*(' + _OPS + r')\s*')
def normalize_sql_operators(text):
    """Setiap operator SQL / karakter khusus diapit tepat satu spasi, lalu spasi dirapatkan.
    Contoh: "1 and 6820=4432--" -> "1 and 6820 = 4432 --";  "1' or '1' = '1" -> "1 ' or ' 1 ' = ' 1"."""
    t = _OP_RE.sub(r' \1 ', str(text))
    return re.sub(r'\s+', ' ', t).strip()

# =====================================================================================
# 2. Data
# =====================================================================================
def load_kaggle(path):
    head = open(path, 'rb').read(4)
    enc = 'utf-16' if head[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig'
    raw = pd.read_csv(path, encoding=enc)
    col = [c for c in raw.columns if c.lower() in ('query', 'sentence')][0]
    raw = raw.rename(columns={col: 'text', 'Label': 'label'})[['text', 'label']]
    df = raw.dropna(subset=['text']).copy(); df['text'] = df.text.astype(str)
    df['norm'] = df.text.map(normalize); df = df[df.norm != '']
    df['label'] = df.label.astype(int)
    k = df.groupby('norm').label.nunique(); conflict = set(k[k > 1].index)
    df = df[~df.norm.isin(conflict)].drop_duplicates('norm').reset_index(drop=True)
    meta = dict(file=os.path.basename(path), sha256=hashlib.sha256(open(path, 'rb').read()).hexdigest(),
                rows_raw=len(raw), rows_clean=len(df))
    return df, meta

def load_sqlmap(path, kaggle_norm):
    S = pd.read_csv(path); S['norm'] = S.payload.map(normalize)
    n_gen = len(S); S = S.drop_duplicates('norm'); n_uni = len(S)
    S = S[~S.norm.isin(set(kaggle_norm))].reset_index(drop=True)      # sama dengan REV9
    return S, dict(generated=n_gen, unique=n_uni, evaluated=len(S))

# =====================================================================================
# 3. Model ML
# =====================================================================================
def vectorizer():
    return TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=20000, sublinear_tf=True,
                           lowercase=False, dtype=np.float32)

MODELS = {   # est: konstruktor; space: ruang tuning (hanya dipakai bila hyperparameter belum ada)
 'KNN': dict(est=lambda: KNeighborsClassifier(algorithm='brute', n_jobs=-1),
             space={'clf__n_neighbors': [3, 5, 7, 9], 'clf__weights': ['uniform', 'distance'], 'clf__metric': ['cosine', 'euclidean']}),
 'SVM': dict(est=lambda: SVC(cache_size=1000, random_state=RS),
             space={'clf__kernel': ['linear', 'rbf'], 'clf__C': [0.1, 1, 10], 'clf__gamma': ['scale']}),
 'Random Forest': dict(est=lambda: RandomForestClassifier(random_state=RS, n_jobs=-1), space=None),
 'XGBoost': dict(est=lambda: XGBClassifier(objective='binary:logistic', eval_metric='logloss', tree_method='hist',
                                           random_state=RS, n_jobs=-1, verbosity=0), space=None, sw=True),
 # ---- baseline linear (Butir 5) ----
 'Logistic Regression': dict(est=lambda: LogisticRegression(solver='liblinear', max_iter=2000, random_state=RS),
             space={'clf__C': [0.1, 1, 10, 100], 'clf__class_weight': [None, 'balanced']}),
 'Naive Bayes': dict(est=lambda: MultinomialNB(),
             space={'clf__alpha': [0.01, 0.1, 0.5, 1.0]}),
}
MAIN = ['KNN', 'SVM', 'Random Forest', 'XGBoost']; LINEAR = ['Logistic Regression', 'Naive Bayes']

def make_pipe(name, params):
    p = Pipeline([('vec', vectorizer()), ('clf', MODELS[name]['est']())]); p.set_params(**params); return p
def fit_kw(name, y): return {'clf__sample_weight': compute_sample_weight('balanced', y)} if MODELS[name].get('sw') else {}

def tune(name, X, y, tr):
    """Protokol tuning REV9: subsampel bertingkat 10.000 data latih, CV 3-fold, macro-F1."""
    sub, _ = train_test_split(tr, train_size=TUNE_SUB, stratify=y[tr], random_state=RS)
    s = GridSearchCV(make_pipe(name, {}), MODELS[name]['space'], scoring='f1_macro',
                     cv=StratifiedKFold(3, shuffle=True, random_state=RS), n_jobs=1, refit=False)
    s.fit(X[sub], y[sub]); return s.best_params_, float(s.best_score_)

def load_hp(path):
    """best_hyperparameters.csv dari run REV9 (Colab); hanya baris split 80:20."""
    hp = {}
    if path and os.path.exists(path):
        h = pd.read_csv(path); h = h[h.Split == '80:20']
        for r in h.itertuples(): hp[r.Algorithm] = {'clf__' + k: v for k, v in json.loads(r.params).items()}
    return hp

# =====================================================================================
# 4. Baseline rule-based
# =====================================================================================
def libinjection_fn():
    try:
        import libinjection
        return lambda s: bool(libinjection.is_sql_injection(s)['is_sqli']), 'libinjection-python ' + _pkgver('libinjection-python')
    except Exception as e:
        log('[PERINGATAN] libinjection tidak tersedia:', e); return None, None

def _pkgver(p):
    try:
        from importlib.metadata import version; return version(p)
    except Exception: return '?'

def _crs_transform(s, tfs):
    for t in tfs:
        if t == 'urlDecodeUni':
            try: s = unquote_plus(s)
            except Exception: pass
        elif t == 'replaceComments': s = re.sub(r'/\*.*?(\*/|$)', ' ', s, flags=re.S)
        elif t == 'removeCommentsChar': s = re.sub(r'/\*|\*/|--|#', '', s)
        elif t == 'removeNulls': s = s.replace('\x00', '')
        elif t == 'removeWhitespace': s = re.sub(r'\s', '', s)
        elif t == 'lowercase': s = s.lower()
        # 'none' dan 'utf8toUnicode' tidak mengubah str Python
    return s

def load_crs(conf_path=None):
    """Mengurai REQUEST-942 dari OWASP CRS. Yang dipakai: aturan @rx dan @detectSQLi yang menyasar ARGS
    (nilai parameter), tanpa 'chain'. Aturan berantai dan aturan khusus header/cookie/filename dicatat
    sebagai dikecualikan."""
    import regex
    txt = open(conf_path).read() if conf_path else urllib.request.urlopen(CRS_URL, timeout=60).read().decode()
    txt = re.sub(r'\\\n\s*', '', txt)
    rules, skipped = [], []
    for line in txt.split('\n'):
        if not line.startswith('SecRule'): continue
        m = re.match(r'SecRule\s+(\S+)\s+"((?:[^"\\]|\\.)*)"\s+"(.*)"\s*$', line)
        if not m: continue
        var, op, act = m.groups()
        rid = re.search(r'id:(\d+)', act); pl = re.search(r"paranoia-level/(\d)", act)
        if not (rid and pl): continue                                   # baris kontrol PL (@lt)
        rid, pl = rid.group(1), int(pl.group(1))
        sev = re.search(r"severity:'(\w+)'", act); sev = sev.group(1) if sev else 'CRITICAL'
        score = {'CRITICAL': 5, 'ERROR': 4, 'WARNING': 3, 'NOTICE': 2}.get(sev, 5)
        targets = var.split('|')
        if 'chain' in act.split(','):  skipped.append((rid, pl, 'chain')); continue
        if 'ARGS' not in targets:      skipped.append((rid, pl, 'bukan ARGS')); continue
        tfs = re.findall(r't:(\w+)', act)
        if op.startswith('@detectSQLi'):
            rules.append(dict(id=rid, pl=pl, score=score, kind='libinjection', tfs=tfs)); continue
        if not op.startswith('@rx'):   skipped.append((rid, pl, op.split()[0])); continue
        pat = op[4:].replace('\\"', '"')
        pat = pat.replace(r'\x{c2}\x{b4}', '´').replace(r'\x{e2}\x80[\x98\x99]', '[‘’]')
        rules.append(dict(id=rid, pl=pl, score=score, kind='rx', tfs=tfs, rx=regex.compile(pat)))
    return rules, skipped, ('local:' + conf_path) if conf_path else CRS_URL

def crs_detector(rules, pl, libinj):
    """True bila skor anomali (jumlah skor aturan yang cocok, PL <= pl) >= 5 (ambang inbound bawaan CRS)."""
    act = [r for r in rules if r['pl'] <= pl and (r['kind'] == 'rx' or libinj is not None)]
    stats = {'timeouts': 0}
    def f(s):
        total = 0
        for r in act:
            v = _crs_transform(s, r['tfs'])
            if r['kind'] == 'libinjection': hit = libinj(v)
            else:
                try: hit = r['rx'].search(v, timeout=1.0) is not None
                except TimeoutError: hit = False; stats['timeouts'] += 1
            if hit:
                total += r['score']
                if total >= 5: return True
        return False
    return f, len(act), stats

# =====================================================================================
# 5. Metrik
# =====================================================================================
def kaggle_metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(Accuracy=100 * accuracy_score(y, p), Precision=100 * precision_score(y, p, zero_division=0),
                Recall=100 * recall_score(y, p), F1=100 * f1_score(y, p), FPR=100 * fp / (fp + tn), FP=int(fp), FN=int(fn))

def mcnemar(a, b, y=None):
    """Uji McNemar eksak (binomial) untuk dua prediksi berpasangan pada baris yang sama.
    Bila y diberikan: benar/salah terhadap y; bila tidak (sqlmap, semua serangan): terdeteksi/tidak."""
    ca, cb = (a == y, b == y) if y is not None else (a == 1, b == 1)
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p

# =====================================================================================
# 6. Main
# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--sqlmap', required=True)
    ap.add_argument('--hp', default=None, help='best_hyperparameters.csv dari run REV9 (dianjurkan)')
    ap.add_argument('--ref', default=None, help='folder hasil REV9 untuk cek reproduksi angka')
    ap.add_argument('--crs', default=None, help='file REQUEST-942 lokal (opsional; default unduh tag ' + CRS_TAG + ')')
    ap.add_argument('--models', default=','.join(MAIN + LINEAR))
    ap.add_argument('--out', default='results_rev10')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    names = [m.strip() for m in a.models.split(',')]
    info = dict(python=platform.python_version(), sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                numpy=np.__version__, pandas=pd.__version__, seed=RS, crs_tag=CRS_TAG)

    # ---- data dan split (identik dengan REV9) ----
    df, dmeta = load_kaggle(a.data); info.update(data=dmeta)
    X, y = df.norm.values, df.label.values
    tr, te = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=RS)
    assert not (set(X[tr]) & set(X[te]))
    S, smeta = load_sqlmap(a.sqlmap, X); info.update(sqlmap=smeta)
    log('Kaggle', dmeta, '| train', len(tr), 'test', len(te), '| sqlmap', smeta)

    # ---- kondisi B: normalisasi operator pada baris yang sama ----
    Xo = np.array([normalize_sql_operators(t) for t in X]); So = S.norm.map(normalize_sql_operators).values
    tr_set = set(Xo[tr])
    coll_te = np.array([t in tr_set for t in Xo[te]])                    # baris uji yang identik dg baris latih setelah normalisasi
    g = pd.DataFrame({'t': Xo, 'y': y}).groupby('t').y.nunique()
    all_o = set(Xo); coll_sq = np.array([t in all_o for t in So])
    leak = dict(test_rows_identical_to_train_after_norm=int(coll_te.sum()),
                label_conflicting_texts_after_norm=int((g > 1).sum()),
                sqlmap_strings_identical_to_kaggle_after_norm=int(coll_sq.sum()),
                sqlmap_with_spaced_equals_before=float(np.mean([' = ' in t for t in S.norm]) * 100),
                sqlmap_with_spaced_equals_after=float(np.mean([' = ' in t for t in So]) * 100),
                kaggle_test_sqli_spaced_equals_before=float(np.mean([' = ' in t for t in X[te][y[te] == 1]]) * 100),
                kaggle_test_sqli_spaced_equals_after=float(np.mean([' = ' in t for t in Xo[te][y[te] == 1]]) * 100))
    info['normalization_check'] = leak; log('normalisasi', leak)

    # ---- hyperparameter ----
    HP = load_hp(a.hp); hp_rows = []
    for n in names:
        if n in HP: src = 'REV9 (' + os.path.basename(a.hp) + ')'; cvs = np.nan
        else:
            if MODELS[n]['space'] is None: sys.exit(f'Hyperparameter {n} tidak ada di --hp; jalankan dengan --hp dari REV9.')
            HP[n], cvs = tune(n, X, y, tr); src = 'dituning ulang (protokol REV9)'
        hp_rows.append(dict(Model=n, params=json.dumps({k[5:]: v for k, v in HP[n].items()}), source=src, cv_macro_f1=cvs))
        log('HP', hp_rows[-1])
    pd.DataFrame(hp_rows).to_csv(f'{a.out}/hyperparameters_rev10.csv', index=False)

    # ---- latih dan prediksi: kondisi A (tanpa) dan B (dengan normalisasi operator) ----
    P = {}; rows = []
    for cond, XX, SS in [('Tanpa normalisasi operator', X, S.norm.values), ('Dengan normalisasi operator', Xo, So)]:
        for n in names:
            p = make_pipe(n, HP[n]); t0 = time.time(); p.fit(XX[tr], y[tr], **fit_kw(n, y[tr])); tt = time.time() - t0
            pk = p.predict(XX[te]); ps = p.predict(SS); P[(cond, n)] = (pk, ps)
            rows.append(dict(Condition=cond, Model=n, Type='ML (TF-IDF char 1-3)', **kaggle_metrics(y[te], pk),
                             sqlmap_DR=100 * ps.mean(), train_s=tt))
            log(cond, '|', n, '|', ', '.join(f'{k}={float(v):.3f}' if isinstance(v, (float, np.floating)) else f'{k}={v}'
                                       for k, v in rows[-1].items() if k not in ('Condition', 'Model', 'Type')))

    # ---- baseline rule-based (dievaluasi pada kedua bentuk input juga) ----
    libinj, libver = libinjection_fn(); info['libinjection'] = libver
    rules, skipped, crs_src = load_crs(a.crs)
    info['crs'] = dict(source=crs_src, rules_used=[(r['id'], r['pl'], r['kind']) for r in rules], rules_skipped=skipped)
    RB = {}
    if libinj: RB['libinjection'] = (libinj, 'Signature (tokenizer)')
    for pl in (1, 2):
        f, nr, st = crs_detector(rules, pl, libinj); RB[f'OWASP CRS {CRS_TAG} PL{pl}'] = (f, f'Rule-based ({nr} aturan, ambang 5)')
        info['crs'][f'PL{pl}_n_rules'] = nr; info['crs'][f'PL{pl}_regex_stats'] = st
    for cond, XX, SS in [('Tanpa normalisasi operator', X, S.norm.values), ('Dengan normalisasi operator', Xo, So)]:
        for n, (f, typ) in RB.items():
            t0 = time.time(); pk = np.array([int(f(s)) for s in XX[te]]); ps = np.array([int(f(s)) for s in SS])
            P[(cond, n)] = (pk, ps)
            rows.append(dict(Condition=cond, Model=n, Type=typ, **kaggle_metrics(y[te], pk), sqlmap_DR=100 * ps.mean(), train_s=np.nan))
            log(cond, '|', n, f"| F1={rows[-1]['F1']:.2f} FPR={rows[-1]['FPR']:.2f} sqlmap={rows[-1]['sqlmap_DR']:.2f} ({time.time() - t0:.0f} s)")
    R = pd.DataFrame(rows); R.to_csv(f'{a.out}/all_results_rev10.csv', index=False)

    # ---- alarm palsu pada input benign yang berupa pernyataan SQL lengkap vs input biasa ----
    STMT = re.compile(r'^\s*(select|insert|update|delete|create|drop|alter|with)\b')
    ben = y[te] == 0; is_stmt = np.array([bool(STMT.search(s)) for s in X[te]])
    fpb = []
    for n in list(names) + list(RB):
        pk = P[('Tanpa normalisasi operator', n)][0]; fpm = ben & (pk == 1)
        fpb.append(dict(Model=n, benign_test=int(ben.sum()), benign_sql_statements=int((ben & is_stmt).sum()),
                        FP=int(fpm.sum()), FP_on_sql_statements=int((fpm & is_stmt).sum()),
                        FPR_sql_statements=100 * fpm[ben & is_stmt].mean(), FPR_other_benign=100 * fpm[ben & ~is_stmt].mean()))
    FB = pd.DataFrame(fpb); FB.to_csv(f'{a.out}/false_alarms_by_input_type.csv', index=False)
    log('\nAlarm palsu menurut jenis input benign (tanpa normalisasi operator):\n' + FB.round(2).to_markdown(index=False))

    # ---- uji berpasangan sebelum vs sesudah + evaluasi tanpa baris yang identik dg data latih ----
    pair = []
    for n in list(names) + list(RB):
        (ka, sa), (kb, sb) = P[('Tanpa normalisasi operator', n)], P[('Dengan normalisasi operator', n)]
        k01, k10, kp = mcnemar(ka, kb, y[te]); s01, s10, sp = mcnemar(sa, sb)
        keep = ~coll_te; keep_s = ~coll_sq
        pair.append(dict(Model=n, kaggle_only_before_correct=k01, kaggle_only_after_correct=k10, kaggle_mcnemar_p=kp,
                         sqlmap_only_before_detected=s01, sqlmap_only_after_detected=s10, sqlmap_mcnemar_p=sp,
                         F1_after_excl_overlap=100 * f1_score(y[te][keep], kb[keep]),
                         FPR_after_excl_overlap=kaggle_metrics(y[te][keep], kb[keep])['FPR'],
                         sqlmap_DR_after_excl_overlap=100 * sb[keep_s].mean()))
    PT = pd.DataFrame(pair); PT.to_csv(f'{a.out}/paired_tests_rev10.csv', index=False)

    # ---- cek reproduksi terhadap angka REV9 ----
    if a.ref and os.path.exists(f'{a.ref}/results_all_splits.csv'):
        ref = pd.read_csv(f'{a.ref}/results_all_splits.csv'); ref = ref[ref.Split == '80:20'].set_index('Algorithm')
        ext = json.load(open(f'{a.ref}/external_sqlmap_summary.json'))['detection_all']
        chk = []
        for n in [m for m in MAIN if m in names]:
            r = R[(R.Condition == 'Tanpa normalisasi operator') & (R.Model == n)].iloc[0]
            chk.append(dict(Model=n, F1_REV9=100 * ref.loc[n, 'F1'], F1_now=r.F1, sqlmap_REV9=ext[n], sqlmap_now=r.sqlmap_DR))
        C = pd.DataFrame(chk); C['match'] = (abs(C.F1_REV9 - C.F1_now) < 0.005) & (abs(C.sqlmap_REV9 - C.sqlmap_now) < 0.005)
        C.to_csv(f'{a.out}/reproduction_check.csv', index=False); info['reproduces_rev9'] = bool(C.match.all())
        log('\nCek reproduksi REV9:\n' + C.round(2).to_markdown(index=False))

    # ---- Tabel A dan B ----
    A = R[R.Condition == 'Tanpa normalisasi operator'].copy()
    A['Kelompok'] = A.Model.map(lambda m: 'Model utama' if m in MAIN else 'Baseline')
    TA = A[['Kelompok', 'Model', 'Type', 'Accuracy', 'Precision', 'Recall', 'F1', 'FPR', 'FP', 'FN', 'sqlmap_DR']].rename(
        columns={'Type': 'Jenis', 'Accuracy': 'Akurasi (%)', 'Precision': 'Presisi (%)', 'Recall': 'Recall (%)', 'F1': 'F1 (%)',
                 'FPR': 'FPR (%)', 'sqlmap_DR': 'Deteksi sqlmap (%)'})
    b0 = R[R.Condition == 'Tanpa normalisasi operator'].set_index('Model'); b1 = R[R.Condition == 'Dengan normalisasi operator'].set_index('Model')
    order = [m for m in list(names) + list(RB)]
    TB = pd.DataFrame({'Model': order,
        'F1 sebelum (%)': b0.loc[order, 'F1'].values, 'F1 sesudah (%)': b1.loc[order, 'F1'].values,
        'FPR sebelum (%)': b0.loc[order, 'FPR'].values, 'FPR sesudah (%)': b1.loc[order, 'FPR'].values,
        'p McNemar (Kaggle)': PT.set_index('Model').loc[order, 'kaggle_mcnemar_p'].values,
        'sqlmap sebelum (%)': b0.loc[order, 'sqlmap_DR'].values, 'sqlmap sesudah (%)': b1.loc[order, 'sqlmap_DR'].values,
        'p McNemar (sqlmap)': PT.set_index('Model').loc[order, 'sqlmap_mcnemar_p'].values})
    TB['Δ sqlmap (pp)'] = TB['sqlmap sesudah (%)'] - TB['sqlmap sebelum (%)']
    TA.to_csv(f'{a.out}/table_A_baselines.csv', index=False); TB.to_csv(f'{a.out}/table_B_operator_norm.csv', index=False)

    def fmt_p(v): return '< 0.001' if v < 0.001 else f'{v:.3f}'
    TBp = TB.copy()
    for c in ['p McNemar (Kaggle)', 'p McNemar (sqlmap)']: TBp[c] = TBp[c].map(fmt_p)
    md = ['## Tabel A. Model utama vs baseline (tanpa normalisasi operator); test set Kaggle 80:20 (n = %d) dan %d string sqlmap\n' % (len(te), len(S)),
          TA.to_markdown(index=False, floatfmt='.2f'),
          '\n\n## Tabel B. Sebelum vs sesudah normalize_sql_operators() pada data latih, test set Kaggle, dan payload sqlmap\n',
          TBp.to_markdown(index=False, floatfmt='.2f'),
          '\n\nCatatan B: hyperparameter sama dengan kondisi tanpa normalisasi; split baris identik. Setelah normalisasi, '
          f"{leak['test_rows_identical_to_train_after_norm']} baris uji identik dengan baris latih dan "
          f"{leak['sqlmap_strings_identical_to_kaggle_after_norm']} string sqlmap identik dengan teks Kaggle "
          '(lihat paired_tests_rev10.csv untuk angka tanpa baris tersebut).']
    open(f'{a.out}/tables_rev10.md', 'w').write('\n'.join(md))
    json.dump(info, open(f'{a.out}/run_info_rev10.json', 'w'), indent=2, default=str)
    print('\n' + '\n'.join(md))
    print('\nUji berpasangan & angka tanpa overlap:\n' + PT.round(4).to_markdown(index=False))

if __name__ == '__main__':
    main()
