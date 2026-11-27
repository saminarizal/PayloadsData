import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd, json
from sklearn.metrics import roc_curve
import sys
R=(sys.argv[1] if len(sys.argv)>1 else 'results/results_sqli').rstrip('/')+'/'; OUT=(sys.argv[2] if len(sys.argv)>2 else 'figures/output').rstrip('/')+'/'
import os; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':9.5,'axes.labelsize':9,
    'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,'savefig.dpi':300,'hatch.linewidth':0.6})
ALG=['KNN','SVM','Random Forest','XGBoost']; LAB=['KNN','SVM','RF','XGBoost']
COL={'KNN':'#0077BB','SVM':'#EE7733','Random Forest':'#009988','XGBoost':'#CC3311'}
HAT={'KNN':'///','SVM':'...','Random Forest':'xxx','XGBoost':'\\\\\\'}
SP=['80:20','70:30','90:10']
res=pd.read_csv(R+'results_all_splits.csv')
def val(a,s,c): return res[(res.Algorithm==a)&(res.Split==s)][c].iloc[0]
def style(ax):
    ax.grid(axis='y',color='#dddddd',lw=0.6); ax.set_axisbelow(True)
    for s in ['top','right']: ax.spines[s].set_visible(False)
x=np.arange(3); w=0.2
# Fig 2: F1 (SQLi) per split
fig,ax=plt.subplots(figsize=(5.9,3.0)); allv=[]
for i,a in enumerate(ALG):
    v=[round(val(a,s,'F1')*100,2) for s in SP]; allv+=v
    ax.bar(x+(i-1.5)*w,v,w*0.92,color=COL[a],hatch=HAT[a],edgecolor='white',linewidth=0.4,label=a)
    for xi,vi in zip(x+(i-1.5)*w,v): ax.text(xi,vi+0.03,f'{vi:.2f}',ha='center',va='bottom',fontsize=8,rotation=90)
lo=np.floor(min(allv)*2)/2-0.5
ax.set_xticks(x); ax.set_xticklabels(SP); ax.set_xlabel('Train:test split'); ax.set_ylabel('F1-score, SQLi class (%)')
ax.set_ylim(lo,100.5); style(ax)
ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(0.5,-0.2),frameon=False)
fig.tight_layout(); fig.savefig(OUT+'fig2_f1.png'); plt.close(fig)
# Fig 3: confusion matrices 80:20
fig,axs=plt.subplots(1,4,figsize=(5.9,1.9))
for ax,a,l in zip(axs,ALG,LAB):
    r=res[(res.Algorithm==a)&(res.Split=='80:20')].iloc[0]; m=np.array([[r.TN,r.FP],[r.FN,r.TP]])
    ax.imshow(m/m.sum(1,keepdims=True),cmap='Blues',vmin=0,vmax=1)
    for i in range(2):
        for j in range(2): ax.text(j,i,f'{m[i,j]:,}',ha='center',va='center',fontsize=9,color='white' if i==j else 'black')
    ax.set_xticks([0,1]); ax.set_xticklabels(['Benign','SQLi']); ax.set_yticks([0,1]); ax.set_yticklabels(['Benign','SQLi'] if a=='KNN' else ['',''])
    ax.set_title(l,fontweight='bold'); ax.set_xlabel('Predicted')
    if a=='KNN': ax.set_ylabel('Actual')
fig.tight_layout(); fig.savefig(OUT+'fig3_cm.png'); plt.close(fig)
# Fig 4: ROC (full + zoom)
roc=pd.read_csv(R+'roc_scores_80_20.csv')
fig,axs=plt.subplots(1,2,figsize=(5.9,2.9))
for a,l in zip(ALG,LAB):
    fpr,tpr,_=roc_curve(roc.y,roc[a]); auc=val(a,'80:20','ROC_AUC')
    for ax in axs: ax.plot(fpr,tpr,color=COL[a],lw=1.3,label=f'{l} (AUC {auc:.4f})')
axs[0].plot([0,1],[0,1],'--',color='#999999',lw=0.8); axs[0].set_title('(a) Full ROC curve')
axs[1].set_xlim(0,0.05); axs[1].set_ylim(0.90,1.001); axs[1].set_title('(b) Zoom: FPR ≤ 0.05')
for ax in axs: ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate'); style(ax); ax.grid(axis='x',color='#dddddd',lw=0.6)
axs[1].legend(loc='lower right',fontsize=8)
fig.tight_layout(); fig.savefig(OUT+'fig4_roc.png'); plt.close(fig)
# Fig 5: stats
cv=pd.read_csv(R+'repeated_cv_folds.csv'); fr=json.load(open(R+'friedman_test.json')); wil=pd.read_csv(R+'wilcoxon_pairwise.csv')
fig=plt.figure(figsize=(5.9,5.4)); gs=fig.add_gridspec(2,2,height_ratios=[1,1.15])
ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1]); ax3=fig.add_subplot(gs[1,:])
for i,a in enumerate(ALG):
    v=cv[cv.Algorithm==a].F1_macro*100
    ax1.scatter(np.full(len(v),i)+np.linspace(-0.15,0.15,len(v)),v,s=10,color=COL[a],alpha=0.7)
    ax1.errorbar(i,v.mean(),yerr=v.std(),fmt='o',color='black',ms=4,capsize=3)
ax1.set_xticks(range(4)); ax1.set_xticklabels(LAB); ax1.set_ylabel('Macro F1-score (%)'); ax1.set_title('(a) Fold-wise macro-F1 (10 folds)'); style(ax1)
rk=[fr['mean_ranks'][a] for a in ALG]
ax2.barh(range(4),rk,color=[COL[a] for a in ALG],hatch=None); ax2.set_yticks(range(4)); ax2.set_yticklabels(LAB); ax2.invert_yaxis()
for i,v in enumerate(rk): ax2.text(v+0.05,i,f'{v:.1f}',va='center',fontsize=8.5)
ax2.set_xlim(0,7.2); ax2.set_xlabel('Friedman mean rank (1 = best)'); ax2.set_xticks([1,2,3,4]); ax2.set_title('(b) Mean rank')
p=fr['friedman_p']; e=int(np.floor(np.log10(p))) if p>0 else 0
ptxt=f"p = {p/10**e:.1f} × 10$^{{{e}}}$" if p<0.001 else f'p = {p:.3f}'
ax2.text(7.1,1.5,f"χ²(3) = {fr['friedman_chi2']:.2f}\n{ptxt}\nW = {fr['kendall_w']:.2f}",ha='right',va='center',fontsize=8)
for ax in (ax1,ax2):
    for s in ['top','right']: ax.spines[s].set_visible(False)
M=np.full((4,4),np.nan); P={}
for _,r in wil.iterrows():
    a,b=r.Comparison.split(' vs '); i,j=ALG.index(a),ALG.index(b); M[j,i]=-r.mean_dF1*100; P[(i,j)]=r.p_holm
lim=max(0.1,np.nanmax(np.abs(M)))
im=ax3.imshow(M,cmap='RdBu',vmin=-lim,vmax=lim)
for i in range(4):
    for j in range(4):
        if i==j: ax3.text(j,i,'–',ha='center',va='center',fontsize=9)
        elif i>j: d=M[i,j]; ax3.text(j,i,f'{d:+.3f}',ha='center',va='center',fontsize=9,color='white' if abs(d)>0.66*lim else 'black')
        else:
            ax3.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,fc='#F2F2F2',ec='white')); pp=P[(i,j)]
            ax3.text(j,i,f'{pp:.4f}'+('' if pp<0.05 else '\nn.s.'),ha='center',va='center',fontsize=8.5)
ax3.set_xticks(range(4)); ax3.set_xticklabels(LAB); ax3.set_yticks(range(4)); ax3.set_yticklabels(LAB)
ax3.set_title('(c) Lower: mean ΔF1 (row − column, pp); upper: Holm p',fontsize=9.5)
cb=fig.colorbar(im,ax=ax3,fraction=0.046,pad=0.04); cb.set_label('ΔF1 (pp)'); cb.ax.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig(OUT+'fig5_stats.png'); plt.close(fig)
# Fig 6: time
fig,axs=plt.subplots(2,1,figsize=(5.9,5.4))
for ax,col,yl,tt,f in [(axs[0],'train_s','Seconds (log scale)','(a) Training time (vectorizer + model)',1),
                       (axs[1],'predict_ms_per_payload','Milliseconds (log scale)','(b) Prediction time per payload',1)]:
    vals=[]
    for i,a in enumerate(ALG):
        v=[val(a,s,col)*f for s in SP]; vals+=v
        ax.bar(x+(i-1.5)*w,v,w*0.92,color=COL[a],hatch=HAT[a],edgecolor='white',lw=0.4,label=a)
        for xi,vi in zip(x+(i-1.5)*w,v): ax.text(xi,vi*1.12,f'{vi:.3g}',ha='center',va='bottom',fontsize=8,rotation=90)
    ax.set_yscale('log'); ax.set_ylabel(yl); ax.set_title(tt); ax.set_xticks(x); ax.set_xticklabels(SP)
    ax.set_ylim(min(vals)/2.5,max(vals)*12); style(ax)
axs[1].set_xlabel('Train:test split'); axs[1].legend(ncol=4,loc='upper center',bbox_to_anchor=(0.5,-0.28),frameon=False)
fig.tight_layout(); fig.savefig(OUT+'fig6_time.png'); plt.close(fig)
# Fig 7: feature importance
fi=pd.read_csv(R+'feature_importance_top20.csv')
fig,axs=plt.subplots(1,2,figsize=(5.9,5.2))
for ax,a in zip(axs,['Random Forest','XGBoost']):
    s=fi[fi.Algorithm==a].sort_values('importance')
    lab=[str(g).replace(' ','␣') for g in s.ngram]
    ax.barh(range(len(s)),s.importance,color=COL[a],hatch=HAT[a],edgecolor='white',lw=0.4)
    ax.set_yticks(range(len(s))); ax.set_yticklabels(lab,fontsize=8.5,family='DejaVu Sans Mono')
    ax.set_title(a,fontweight='bold'); ax.set_xlabel('Importance'); style(ax); ax.grid(axis='x',color='#dddddd',lw=0.6); ax.grid(axis='y',visible=False)
fig.tight_layout(); fig.savefig(OUT+'fig7_feat.png'); plt.close(fig)
# Fig 8: external sqlmap by technique
ext=json.load(open(R+'external_sqlmap_summary.json'))
tech=['boolean-based blind','error-based','inline query','stacked queries','time-based blind','UNION query']
A=[[ext['by_technique'][t][a] for a in ALG] for t in tech]; rl=[f"{t} (n = {ext['by_technique'][t]['n']:,})" for t in tech]
A.append([ext['detection_level1'][a] for a in ALG]); rl.append(f"Default level 1 (n = {ext['level1']})")
A.append([ext['detection_all'][a] for a in ALG]); rl.append(f"All sqlmap strings (n = {ext['evaluated']:,})")
A.append([ext['detection_spaced'][a] for a in ALG]); rl.append("All strings, operators spaced")
M=np.array(A)
fig,ax=plt.subplots(figsize=(5.9,3.6))
im=ax.imshow(M,cmap='YlGnBu',vmin=75,vmax=100,aspect='auto')
for i in range(M.shape[0]):
    for j in range(4):
        v=M[i,j]; ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=8.5,color='white' if v>=89 else 'black',fontweight='bold' if i>=M.shape[0]-2 else 'normal')
ax.set_xticks(range(4)); ax.set_xticklabels(LAB); ax.xaxis.tick_top(); ax.set_yticks(range(len(rl))); ax.set_yticklabels(rl)
ax.axhline(len(tech)-0.5,color='black',lw=0.8); ax.axhline(len(tech)+0.5,color='black',lw=0.8); ax.axhline(len(tech)+1.5,color='black',lw=0.8,ls=':')
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
cb=fig.colorbar(im,ax=ax,fraction=0.04,pad=0.02,extend='min'); cb.set_label('Classified as SQLi (%)')
fig.tight_layout(); fig.savefig(OUT+'fig8_external.png'); plt.close(fig)
print('figs ok')
