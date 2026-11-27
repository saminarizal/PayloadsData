# Architecture and flow of the SQLi detection system (training, online detection, validation)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family': 'DejaVu Sans', 'savefig.dpi': 300})
W, H = 6.3, 5.3
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
C = {'train': ('#0077BB', '#E1EEF8'), 'online': ('#009988', '#DDF2EF'), 'valid': ('#EE7733', '#FDEBDD'), 'grey': ('#555555', '#F2F2F2')}
def lane(y0, h, key, label):
    col, bg = C[key]
    ax.add_patch(FancyBboxPatch((0.05, y0), W - 0.1, h, boxstyle='round,pad=0,rounding_size=0.08', fc=bg, ec='none', alpha=0.55))
    ax.add_patch(FancyBboxPatch((0.05, y0), 0.30, h, boxstyle='round,pad=0,rounding_size=0.08', fc=col, ec='none'))
    ax.text(0.20, y0 + h / 2, label, rotation=90, ha='center', va='center', color='white', fontsize=8.5, fontweight='bold')
def box(x, y, w, h, key, title, body, num=None):
    col, bg = C[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.06', fc='white', ec=col, lw=1.2, zorder=3))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.24), w, 0.24, boxstyle='round,pad=0,rounding_size=0.06', fc=col, ec=col, lw=1.2, zorder=4))
    ax.text(x + w / 2, y + h - 0.12, title, ha='center', va='center', color='white', fontsize=8, fontweight='bold', zorder=5)
    ax.text(x + w / 2, y + (h - 0.24) / 2, body, ha='center', va='center', fontsize=8, color='#222222', zorder=5, linespacing=1.2)
    return (x, y, w, h)
def arrow(p, q, col='#444444', ls='-', rad=0.0, text=None, tx=0, ty=0):
    a = FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=10, color=col, lw=1.1, linestyle=ls, connectionstyle=f'arc3,rad={rad}', zorder=6)
    ax.add_patch(a)
    if text: ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2 + ty, text, fontsize=8, color=col, ha='center', va='center', style='italic', zorder=7,
                     bbox=dict(fc='white', ec='none', pad=0.5))
def R(b): return (b[0] + b[2], b[1] + (b[3] - 0.24) / 2)
def L(b): return (b[0], b[1] + (b[3] - 0.24) / 2)
def T(b): return (b[0] + b[2] / 2, b[1] + b[3])
def Bm(b): return (b[0] + b[2] / 2, b[1])
bw, bh, gap = 1.08, 0.98, 0.13
xs = [0.48 + i * (bw + gap) for i in range(4)]
# lanes
lane(3.82, 1.40, 'train', 'TRAINING\n(offline)')
lane(2.05, 1.62, 'online', 'DETECTION\n(online, WAF)')
lane(0.08, 1.82, 'valid', 'VALIDATION')
# training lane
yt = 4.02
t1 = box(xs[0], yt, bw, bh, 'train', 'Kaggle dataset', 'Kaggle SQLi file\n30,919 strings\nBenign / SQLi')
t2 = box(xs[1], yt, bw, bh, 'train', 'Clean', 'Normalise, drop\nempty, conflicting\n& duplicates\n→ 30,802 unique')
t3 = box(xs[2], yt, bw, bh, 'train', 'Split & tune', 'Stratified 80:20\n(70:30, 90:10)\n3-fold search\nmacro-F1')
t4 = box(xs[3], yt, bw, bh, 'train', 'Fitted pipeline', 'TF-IDF + model\nfitted on\ntraining data\nonly')
for a, b in [(t1, t2), (t2, t3), (t3, t4)]: arrow(R(a), L(b))
# online lane
yo = 2.25; bh2 = 1.18
o1 = box(xs[0], yo, bw, bh2, 'online', 'Input', "Form field,\nURL parameter,\nor cookie\n1' or '1'='1")
o2 = box(xs[1], yo, bw, bh2, 'online', 'Normaliser', 'URL decode\nHTML unescape\nlower-case\ncollapse spaces')
o3 = box(xs[2], yo, bw, bh2, 'online', 'TF-IDF', 'Char 1–3-grams\n20,000 features\nL2-normalised\n(frozen)')
o4 = box(xs[3], yo, bw, bh2, 'online', 'Classifier', 'KNN · SVM\nRF · XGBoost\n→ Benign or SQLi')
for a, b in [(o1, o2), (o2, o3), (o3, o4)]: arrow(R(a), L(b))
# decision outputs (right edge)
dx = xs[3] + bw + 0.16; dw = W - 0.12 - dx
ax.add_patch(FancyBboxPatch((dx, yo + 0.62), dw, 0.54, boxstyle='round,pad=0,rounding_size=0.05', fc='#CC3311', ec='none', zorder=3))
ax.text(dx + dw / 2, yo + 0.89, 'SQLi\n→ block\nand log', ha='center', va='center', fontsize=8, color='white', fontweight='bold', zorder=5)
ax.add_patch(FancyBboxPatch((dx, yo + 0.02), dw, 0.54, boxstyle='round,pad=0,rounding_size=0.05', fc='#009988', ec='none', zorder=3))
ax.text(dx + dw / 2, yo + 0.29, 'Benign\n→ app\nand DB', ha='center', va='center', fontsize=8, color='white', fontweight='bold', zorder=5)
arrow((xs[3] + bw, yo + 0.62), (dx, yo + 0.89), col='#CC3311')
arrow((xs[3] + bw, yo + 0.40), (dx, yo + 0.29), col='#009988')
# deploy arrows
arrow(Bm(t4), T(o4), col='#0077BB', ls='--', text='deploy', tx=0.30)
arrow((xs[2] + bw / 2, yt), T(o3), col='#0077BB', ls='--')
# validation lane
yv = 0.28; bh3 = 1.10
v1 = box(xs[0], yv, bw, bh3, 'valid', 'Kaggle test set', '6,161 held-out\npayloads (80:20)\n+ 2 × 5-fold\nCV folds')
v2 = box(xs[1], yv, bw, bh3, 'valid', 'sqlmap 1.10.11', '366 templates\n× 53 boundaries\n→ 15,332 new\nSQLi strings')
v3 = box(xs[2], yv, bw, bh3, 'valid', 'Normalisation', 'Operators spaced\nin train and test:\n6820=4432\n→ 6820 = 4432')
v4 = box(xs[3], yv, bw, bh3, 'valid', 'Measures', 'F1, FPR, FNR,\nROC-AUC,\nFriedman, Wilcoxon,\nMcNemar tests,\ndetection rate')
arrow(R(v2), L(v3))
for b in (v1, v2, v3): arrow(T(b), Bm(o2), col='#EE7733', ls=':', rad=0.0)
arrow(Bm(o4), T(v4), col='#EE7733', text='predictions', tx=0.42)
fig.savefig(__import__('os').path.join(__import__('sys').argv[1] if len(__import__('sys').argv)>1 else 'figures/output', 'fig_arch.png'), dpi=300)
print('ok')
