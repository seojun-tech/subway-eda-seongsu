import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fam=[f.name for f in fm.fontManager.ttflist if 'CJK' in f.name and 'Sans' in f.name]
plt.rcParams['font.family']='Noto Sans CJK KR' if 'Noto Sans CJK KR' in fam else fam[0]
BG='#fcfcfb'; T1='#0b0b0b'; T2='#52514e'; GR='#e9e8e4'; AX='#b8b7b0'
PAL=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4']
def base(title,sub,figsize=(10,5.6)):
    fig,ax=plt.subplots(figsize=figsize,dpi=200); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['left','bottom']: ax.spines[sp].set_color(AX)
    ax.tick_params(colors=T2); ax.grid(axis='y',color=GR,lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title,loc='left',fontsize=15,fontweight='bold',color=T1,pad=26)
    ax.text(0,1.035,sub,transform=ax.transAxes,fontsize=9.5,color=T2)
    return fig,ax
def labels_right(ax,xs,series,colors,fmt):
    # 오른쪽 끝 직접 라벨, 겹치면 위아래로 밀기
    ends=sorted([(s[-1],n,c) for n,s,c in zip(series.keys(),series.values(),colors)])
    lo,hi=ax.get_ylim(); gap=(hi-lo)*0.045; placed=[]
    for v,n,c in ends:
        y=v
        if placed and y-placed[-1]<gap: y=placed[-1]+gap
        placed.append(y)
        ax.annotate(fmt(n,v),(xs[-1],v),xytext=(xs[-1]+0.15,y),textcoords='data',va='center',fontsize=10.5,color=T1,fontweight='bold',
                    arrowprops=dict(arrowstyle='-',color=c,lw=1) if abs(y-v)>gap*0.3 else None)
