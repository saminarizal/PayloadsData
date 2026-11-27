import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import sys
SQLI_ROWS=''
plt.rcParams.update({'font.family':'DejaVu Sans','savefig.dpi':300})
W,H=5.9,7.52
fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis('off')
PH=[('PHASE I\nData\npreparation','#2F6690','#E3EEF7'),('PHASE II\nModelling','#3A7D44','#E5F2E7'),('PHASE III\nEvaluation and\nvalidation','#B5651D','#FBEBDD')]
ST=[ # phase, no, title, section, detail lines, output
 (0,1,'Data collection','2.1',['Kaggle SQL Injection Dataset, Modified_SQL_Dataset.csv','Label 0 → Benign (19,537); label 1 → SQLi (11,382)'],'30,919 labelled strings'),
 (0,2,'Normalisation and deduplication','2.2',['URL and HTML-entity decoding, lower-casing, whitespace collapsing','Empty, label-conflicting, and duplicate payloads removed'],'30,802 unique payloads'),
 (0,3,'Stratified splitting','2.2',['80:20, 70:30, and 90:10, seed 42'],'No identical payload in train and test'),
 (1,4,'Feature extraction','2.3',['Character 1–3-gram TF-IDF, 20,000 features','Fitted on training data only, inside the pipeline'],'Sparse payload vectors'),
 (1,5,'Training and tuning','2.4',['KNN  ·  SVM  ·  Random Forest  ·  XGBoost','Grid or randomised search, 3-fold CV, macro-F1'],'Four tuned models per split'),
 (2,6,'Internal evaluation and baselines','2.5–2.6, 2.8',['Hold-out metrics, FPR/FNR, ROC-AUC, run time; ablation; 2 × 5-fold CV','Baselines: logistic regression, naive Bayes, libinjection, OWASP CRS'],'Ranking with statistical support'),
 (2,7,'External test and operator normalisation','2.7–2.8',['sqlmap 1.10.11: 366 templates × 53 boundaries → 15,332 strings','Spacing check; retraining with normalised operators (McNemar)'],'Detection rate outside the dataset'),
 (2,8,'Analysis and comparison','3',['Errors, cost, influential n-grams, class contents','Comparison with studies on the same Kaggle file'],'Recommendation for WAF use'),
]
lx,lw=0.08,0.95; cx=1.13; cw=W-cx-0.08
top=H-0.08; ch=0.80; gap=0.10; pg=0.16
y=top; pos={}
for p in range(3):
    n=[s for s in ST if s[0]==p]; hh=len(n)*ch+(len(n)-1)*gap
    col,bg=PH[p][1],PH[p][2]
    ax.add_patch(FancyBboxPatch((lx,y-hh),lw,hh,boxstyle='round,pad=0,rounding_size=0.08',fc=col,ec='none'))
    ax.text(lx+lw/2,y-hh/2,PH[p][0],ha='center',va='center',color='white',fontsize=8,fontweight='bold',linespacing=1.3)
    for s in n:
        yb=y-ch
        ax.add_patch(FancyBboxPatch((cx,yb),cw,ch,boxstyle='round,pad=0,rounding_size=0.07',fc=bg,ec=col,lw=1.0))
        ax.add_patch(Circle((cx+0.24,yb+ch/2),0.16,fc=col,ec='white',lw=1.2,zorder=3))
        ax.text(cx+0.24,yb+ch/2,str(s[1]),ha='center',va='center',color='white',fontsize=9.5,fontweight='bold',zorder=4)
        tx=cx+0.50
        ax.text(tx,yb+ch-0.07,s[2],ha='left',va='top',fontsize=9,fontweight='bold',color='#222222')
        ax.text(cx+cw-0.08,yb+ch-0.07,'Sec. '+s[3],ha='right',va='top',fontsize=8,color=col,fontweight='bold')
        ax.text(tx,yb+ch-0.27,'\n'.join(s[4]),ha='left',va='top',fontsize=8,color='#333333',linespacing=1.25)
        ax.text(cx+cw-0.08,yb+0.06,'→ '+s[5],ha='right',va='bottom',fontsize=8,color=col,style='italic')
        pos[s[1]]=(yb,yb+ch)
        y=yb-gap
    y=y+gap-pg
for k in range(1,8):
    a=pos[k][0]; b=pos[k+1][1]
    ax.annotate('',xy=(cx+0.24,b),xytext=(cx+0.24,a),arrowprops=dict(arrowstyle='-|>',color='#555555',lw=1.0,mutation_scale=9),zorder=5)
fig.savefig(__import__('os').path.join(__import__('sys').argv[1] if len(__import__('sys').argv)>1 else 'figures/output', 'fig1_flow.png'),dpi=300); print('ok', y)
