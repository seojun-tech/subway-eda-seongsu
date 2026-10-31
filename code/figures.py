# 그림 1~9: 1~3단계 결과(out/*.csv)와 캐시로 그림
import numpy as np, pandas as pd
from common import *
from style import *

def O(f, **k): return pd.read_csv(f'{OUT}/{f}', encoding='utf-8-sig', **k)
def save(fig, name): fig.tight_layout(); fig.savefig(f'{OUT}/{name}', facecolor=BG); plt.close(fig); print(name)
def covid(ax, x0, x1, top=True):
    ax.axvspan(x0, x1, color='#e9e8e4', zorder=0)
    ax.text((x0 + x1) / 2, 0.97 if top else 0.05, '코로나 구간 (비교 제외)', transform=ax.get_xaxis_transform(), ha='center', va='top' if top else 'bottom', fontsize=10.5, color=T2)
def line(ax, xs, ys, c, ls='-', lw=2.6, ms=6.5):
    ax.plot(xs, ys, color=c, lw=lw, ls=ls, marker='o', ms=ms, mec='white', mew=1.5, zorder=3)

def fig1():
    s = O('성수_점유율_평일주말공휴일.csv'); xs = s.연도.values
    fig, ax = base('성수역 주말 점유율은 7년 새 3.4배, 2025년부터 평일을 넘었다',
                   '각 해 1~6월 하루 평균, 공휴일·5월 1일 제외, 분모 587개 공통 역 / 해마다 평일 vs 주말 Mann-Whitney: 2019~2023 평일>주말, 2024 차이 없음, 2025·2026 주말>평일')
    covid(ax, 2019.5, 2022.5, top=False)
    for k, c in [('평일', PAL[0]), ('주말', PAL[1])]:
        line(ax, xs, s[k], c); v0, v1 = s[k].iloc[0], s[k].iloc[-1]
        ax.text(2026.2, v1, f'{k} ×{v1 / v0:.1f}', va='center', fontsize=13, fontweight='bold', color=T1)
        up = k == '평일'
        ax.annotate(f'{v0:.2f}%', (2019, v0), xytext=(0, 12 if up else -22), textcoords='offset points', ha='center', fontsize=11, color=T1)
        ax.annotate(f'{v1:.2f}%', (2026, v1), xytext=(0, -24 if up else 12), textcoords='offset points', ha='center', fontsize=11, color=T1)
    ax.set_ylim(0, 1.0); ax.set_xlim(2018.6, 2027.0); ax.set_xticks(xs); ax.set_ylabel('서울 지하철 전체 대비 성수역 승하차 점유율 (%)', color=T2)
    save(fig, 'fig1_성수_점유율.png')

def fig2():
    s = O('Q1_주말점유율_2019지수.csv', index_col=0); xs = s.index.values
    fig, ax = base('2019년 대비 주말 점유율: 성수 3.4배, 서울숲·뚝섬 약 1.6배, 건대입구 0.9배',
                   '주말 점유율, 2019년 = 100 / 각 해 1~6월, 공휴일 제외 / 2019 vs 2023~2026 각 해 Mann-Whitney + Holm, 모두 p<0.001')
    covid(ax, 2019.5, 2022.5); ax.axhline(100, color=AX, ls='--', lw=1)
    cols = {'성수': PAL[0], '서울숲': PAL[1], '뚝섬': PAL[2], '건대입구': PAL[3]}
    for g, c in cols.items(): line(ax, xs, s[g], c)
    ax.set_ylim(0, 400); ax.set_xlim(2018.6, 2027.6); ax.set_xticks(xs); ax.set_ylabel('2019년 = 100', color=T2)
    labels_right(ax, xs, {g: s[g].values for g in cols}, list(cols.values()), lambda n, v: f'{n} {v:.0f}')
    save(fig, 'fig2_성수와_주변역.png')

def fig3():
    """질문 2 가설 B: 용산·명동도 오르지만 성수가 더 빨리 큰다 (2023 = 100)"""
    import s1_subway as s1
    x = s1.daily(s1.load()); x = x[(x.구분 == '주말') & (x.연도 >= 2023)]
    cols = {'성수': PAL[0], '명동': PAL[1], '용산': PAL[2]}
    m = x.groupby('연도')[list(cols) + ['전체']].mean(); sh = m[list(cols)].div(m['전체'], axis=0); idx = sh / sh.loc[2023] * 100
    pg = O('Q2_Page_지하철.csv').set_index('대상')
    ry, rm = pg.loc['성수÷용산'], pg.loc['성수÷명동']
    fig, ax = base('용산·명동도 오르지만, 성수가 가장 빨리 큰다',
                   f'주말 점유율, 2023년 = 100 / 각 해 1~6월, 공휴일 제외 / 성수÷용산 {ry["2023"]:.2f} → {ry["2026"]:.2f}, 성수÷명동 {rm["2023"]:.2f} → {rm["2026"]:.2f}'
                   f' (같은 달끼리 Page 검정 p<0.001)')
    xs = idx.index.values; ax.axhline(100, color=AX, ls='--', lw=1)
    for g, c in cols.items(): line(ax, xs, idx[g], c)
    ax.set_ylim(80, 190); ax.set_xlim(2022.8, 2026.75); ax.set_xticks(xs); ax.set_ylabel('2023년 = 100', color=T2)
    labels_right(ax, xs, {g: idx[g].values for g in cols}, list(cols.values()), lambda n, v: f'{n} {v:.0f}')
    save(fig, 'fig3_최근추세.png')
    return idx.round(0)

def fig4():
    p = O('Q3_주말_시간대별_도착비중_2026.csv', index_col=0); q = O('Q3_생활이동_2026상반기.csv').set_index('지표')
    fig, ax = base('주말의 성수동은 낮에, 화양동(건대입구)은 밤에 사람이 도착한다',
                   '2026년 1~6월 주말 49일 합계, 시간대별 도착 비중 / 서울시·KT 생활이동(추정치) / 같은 날짜끼리 Wilcoxon: 낮·저녁 비중 모두 p<0.001')
    ax.axvspan(9.5, 16.5, color='#e9e8e4', zorder=0); ax.axvspan(17.5, 23.5, color='#f3f2ef', zorder=0)
    a, b = q.loc['주말 10~16시 비중'], q.loc['주말 18~23시 비중']
    ax.text(13, 10.5, f'10~16시 도착\n성수 {a.성수동:.0f}% · 화양 {a.화양동:.0f}%', ha='center', va='center', fontsize=11, color=T2)
    ax.text(20.5, 10.5, f'18~23시 도착\n성수 {b.성수동:.0f}% · 화양 {b.화양동:.0f}%', ha='center', va='center', fontsize=11, color=T2)
    h = p.index.values
    ax.plot(h, p['성수동'], color=PAL[0], lw=2.6, marker='o', ms=5, mec='white', label='성수동', zorder=3)
    ax.plot(h, p['화양동'], color=PAL[1], lw=2.6, marker='o', ms=5, mec='white', label='화양동', zorder=3)
    ax.plot(h, p['서울 전체'], color='#8a8984', lw=2, ls='--', label='서울 전체', zorder=2)
    for r, c in [('성수동', PAL[0]), ('화양동', PAL[1])]:
        k = p[r].idxmax(); ax.annotate(f'{r} 최고 {k}시', (k, p[r].max()), xytext=(0, 14), textcoords='offset points', ha='center', fontsize=11.5, fontweight='bold', color=T1)
    ax.legend(frameon=False, loc='upper left', fontsize=11); ax.set_xlim(-0.5, 23.5); ax.set_ylim(0, 11.5); ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel('도착 시각 (시)', color=T2); ax.set_ylabel('하루 도착 인원 중 비중 (%)', color=T2)
    save(fig, 'fig4_성수_화양_시간대.png')

def fig5():
    f = O('금요일효과.csv'); xs = np.arange(len(f))
    fig, ax = base('금요일에 더 붐비는 정도: 2019년 성수는 평범했고, 2024년부터 상위권이다',
                   f'평일 승하차, 금요일 평균 ÷ 월~목 평균 / 각 해 1~6월, 공휴일 제외, 평일 평균 1,000명 이상 역 {f.역수.min()}~{f.역수.max()}개 중 순위\n'
                   '성수 점유율 금 vs 월~목 (Mann-Whitney): 2019 금<월목, 2023 차이 없음(p=0.07), 2024~2026 금>월목 (p<0.001)', figsize=(10, 5.6))
    ax.axhline(1, color=AX, lw=1.2); ax.text(xs[-1] + 0.1, 1.003, '월~목과 같음', fontsize=10, color=T2, ha='center')
    line(ax, xs, f.성수_금요일증가율, PAL[0]); line(ax, xs, f.전체역_중앙값, '#8a8984', ls='--', lw=2)
    for x, (v, pc) in enumerate(zip(f.성수_금요일증가율, f.성수_상위퍼센트)):
        ax.annotate(f'{v:.3f}\n(상위 {pc:.1f}%)', (x, v), xytext=(0, -36 if x == 0 else 10), textcoords='offset points', ha='center', fontsize=10.5, color=T1)
    ax.text(xs[-1] + 0.12, f.성수_금요일증가율.iloc[-1], '성수역', va='center', fontsize=12, fontweight='bold', color=T1)
    ax.text(xs[-1] + 0.12, f.전체역_중앙값.iloc[-1], '전체 역 중앙값', va='center', fontsize=11, color=T2)
    ax.axvline(0.5, color=AX, ls=':', lw=1); ax.text(0.53, 0.97, '2020~2022\n코로나 구간 생략', transform=ax.get_xaxis_transform(), fontsize=9.5, color=T2, va='top')
    ax.set_xticks(xs); ax.set_xticklabels(f.연도); ax.set_xlim(-0.4, len(f) - 0.1); ax.set_ylim(0.99, 1.115); ax.set_ylabel('금요일 ÷ 월~목 (배)', color=T2)
    ax.set_title(ax.get_title('left'), loc='left', fontsize=15, fontweight='bold', color=T1, pad=46)
    save(fig, 'fig5_금요일효과.png')

def fig6():
    c = O('Q3_생활이동_상반기_2023_2026.csv'); tr = O('Q3_생활이동_성수와차이_추세.csv').set_index('지표')
    a, b = tr.loc['평일 출근 비중'], tr.loc['주말 기타 비중']
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.4), dpi=200); fig.patch.set_facecolor(BG)
    fig.suptitle('성수의 "출근 동네" 성격은 서울 평균보다 빠르게 옅어지고 있다', x=0.06, ha='left', fontsize=15, fontweight='bold', color=T1, y=0.97)
    fig.text(0.06, 0.885, f'각 해 1~6월 합계 / 서울시·KT 생활이동(추정치), 도착지 기준 / 성수−서울 차이의 일별 값 × 연도 Spearman: '
             f'평일 출근 ρ={a.rho:.2f} (p<0.001), 주말 기타 ρ={b.rho:.2f} (p={b.p:.2f})', fontsize=9.5, color=T2)
    for ax, (name, ttl, lim) in zip(axs, [('평일 출근 비중', '평일: 도착 인원 중 출근 목적 (%)', (0, 40)), ('주말 기타 비중', '주말: 도착 인원 중 기타(여가 추정) 목적 (%)', (40, 80))]):
        ax.set_facecolor(BG); [ax.spines[s].set_visible(False) for s in ['top', 'right']]; [ax.spines[s].set_color(AX) for s in ['left', 'bottom']]
        ax.tick_params(colors=T2); ax.grid(axis='y', color=GR, lw=0.8); ax.set_axisbelow(True); ax.set_title(ttl, loc='left', fontsize=12, color=T1)
        s = c[c.지표 == name]; xs = s.연도.values
        for r, col, ls, bold in [('성수동', PAL[0], '-', True), ('화양동', PAL[1], '-', False), ('서울 전체', '#8a8984', '--', False)]:
            line(ax, xs, s[r], col, ls=ls, lw=2.6 if bold else 2, ms=6)
            ax.text(xs[-1] + 0.12, s[r].iloc[-1], f'{r} {s[r].iloc[0]:.1f}→{s[r].iloc[-1]:.1f}', va='center', fontsize=11, color=T1, fontweight='bold' if bold else 'normal')
        ax.set_ylim(*lim); ax.set_xlim(2022.7, 2027.5); ax.set_xticks(xs)
    fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig(f'{OUT}/fig6_성수_성격변화.png', facecolor=BG); plt.close(fig); print('fig6')

def fig7():
    t = O('성수_주말평일비율_순위.csv'); xs = np.arange(len(t))
    fig, ax = base(f'주말에 비던 출근 역 성수, 2026년엔 주말 비율 상위 {t.상위pct.iloc[-1]:.0f}%',
                   '역별 주말 하루 평균 ÷ 평일 하루 평균 (승하차) / 각 해 1~6월, 공휴일 제외, 평일 평균 승차 1,000명 이상 역 / 순위는 비율이 높은 순')
    line(ax, xs, t.성수, PAL[0], ms=7); line(ax, xs, t.중앙값, '#8a8984', ls='--', lw=2)
    for x, (v, k, n) in enumerate(zip(t.성수, t.순위, t.역수)):
        ax.annotate(f'{v:.2f}\n{n}개 중 {k}위', (x, v), xytext=(0, -34 if x <= 1 else 14), textcoords='offset points', ha='center', fontsize=10.5, color=T1)
    ax.text(xs[-1] + 0.12, t.성수.iloc[-1], '성수역', va='center', fontsize=12, fontweight='bold', color=T1)
    ax.text(xs[-1] + 0.12, t.중앙값.iloc[-1], '전체 역 중앙값', va='center', fontsize=11, color=T2)
    ax.axvline(0.5, color=AX, ls=':', lw=1); ax.text(0.53, 0.95, '2020~2022\n코로나 구간 생략', fontsize=9.5, color=T2, va='top')
    ax.set_xticks(xs); ax.set_xticklabels(t.연도); ax.set_xlim(-0.4, len(t) - 0.2); ax.set_ylim(0.25, 1.0); ax.set_ylabel('주말 ÷ 평일 (배)', color=T2)
    save(fig, 'fig7_성수_주말평일비율.png')

def fig8():
    """질문 1: 2026 주말에 성수동·화양동으로 온 사람의 출발 지역 (같은 12일 평균)"""
    t = O('Q1_출발지_36일_지표.csv', dtype={'연': str}); w = t[t.요일.isin(['토', '일']) & (t.연 == '2026')]
    g = w.groupby('도착')[['성동광진', '서울다른구', '서울밖', '중앙거리km']].mean(); rows = ['화양동', '성수동']
    fig, ax = base('성수동 주말 방문객은 화양동(건대입구)보다 멀리서 온다',
                   '2026년 1~6월 주말(토·일) 12일, 도착 이동의 출발 지역 비율 평균 / 같은 동네 안 이동 제외 / 서울시·KT 생활이동(출발-도착지, 추정치)\n'
                   '같은 날짜끼리 Wilcoxon (12쌍): 서울 밖·서울 다른 구 비율과 이동거리는 성수가 높고 성동·광진 비율은 낮음, 12일 모두 같은 방향 (p<0.001)', figsize=(10, 4.4))
    ax.grid(False); ys = np.arange(len(rows)); left = np.zeros(len(rows))
    for k, lab, c in [('성동광진', '성동·광진', '#b8b7b0'), ('서울다른구', '서울 다른 구', PAL[0]), ('서울밖', '서울 밖', PAL[1])]:
        v = g.loc[rows, k].values; ax.barh(ys, v, left=left, color=c, height=0.6, label=lab)
        for y, (l, x) in enumerate(zip(left, v)): ax.text(l + x / 2, y, f'{x:.1f}%', ha='center', va='center', fontsize=12, color=T1 if c == '#b8b7b0' else 'white', fontweight='bold')
        left += v
    for y, r in enumerate(rows): ax.text(101.5, y, f'중앙 이동거리\n{g.loc[r, "중앙거리km"]:.1f}km', va='center', fontsize=11, color=T1)
    ax.set_yticks(ys); ax.set_yticklabels(rows, fontsize=13, color=T1); ax.set_xlim(0, 118); ax.set_xticks([0, 25, 50, 75, 100])
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    ax.legend(frameon=False, ncol=3, loc='upper left', bbox_to_anchor=(0, -0.1), fontsize=11)
    ax.set_title(ax.get_title('left'), loc='left', fontsize=15, fontweight='bold', color=T1, pad=46)
    save(fig, 'fig8_출발지역.png'); return g

def fig10():
    """질문 1 가설 B·C: 2023→2026 같은 달·같은 요일 짝 12쌍 (주말): 서울 도착 중 몫, 연계 이동 비율"""
    t = O('Q1_출발지_36일_지표.csv', dtype={'연': str, '월': str}); w = t[t.요일.isin(['토', '일'])]
    panels = [('서울 도착 이동 중 몫 (%)', [('성수동', '서울몫', '성수동', PAL[0]), ('화양동', '서울몫', '화양동', PAL[1])]),
              ('나간 이동 중 연계 비율 (%)', [('성수동', '상대로간비율', '성수동 → 화양동(건대)', PAL[0]), ('성수동', '성수2가_서울숲비율', '성수역 쪽 → 서울숲(성수1가1동)', PAL[2])])]
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.4), dpi=200); fig.patch.set_facecolor(BG)
    fig.suptitle('2023→2026: 성수는 서울 안에서 몫을 키웠지만, 주변으로 이어지는 이동은 줄었다', x=0.06, ha='left', fontsize=15, fontweight='bold', color=T1, y=0.97)
    fig.text(0.06, 0.885, '각 해 1~6월 달마다 토·일 1일씩, 같은 달·같은 요일끼리 짝 12쌍 (가는 선 = 한 쌍, 굵은 선 = 평균) / 서울시·KT 생활이동(출발-도착지, 추정치) / Wilcoxon 부호순위', fontsize=9.5, color=T2)
    for ax, (ttl, items) in zip(axs, panels):
        ax.set_facecolor(BG); [ax.spines[k].set_visible(False) for k in ['top', 'right']]; [ax.spines[k].set_color(AX) for k in ['left', 'bottom']]
        ax.tick_params(colors=T2); ax.grid(axis='y', color=GR, lw=0.8); ax.set_axisbelow(True); ax.set_title(ttl, loc='left', fontsize=12, color=T1)
        for dest, k, lab, c in items:
            a = w[(w.도착 == dest) & (w.연 == '2023')].sort_values(['요일', '월'])[k].values; b = w[(w.도착 == dest) & (w.연 == '2026')].sort_values(['요일', '월'])[k].values
            for u, v in zip(a, b): ax.plot([0, 1], [u, v], color=c, alpha=0.22, lw=1)
            ax.plot([0, 1], [a.mean(), b.mean()], color=c, lw=3, marker='o', ms=8, mec='white', mew=1.5, zorder=3)
            from scipy import stats
            p = stats.wilcoxon(b, a).pvalue; up = int((b > a).sum())
            ax.text(1.06, b.mean(), f'{lab}\n{a.mean():.2f} → {b.mean():.2f}  ({up}/12 늘어남, p={p:.3f})' if p >= 0.001 else f'{lab}\n{a.mean():.2f} → {b.mean():.2f}  ({up}/12 늘어남, p<0.001)',
                    va='center', fontsize=10, color=T1, fontweight='bold')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['2023', '2026'], fontsize=12); ax.set_xlim(-0.15, 2.05)
    axs[0].set_ylim(0, 1.3); axs[1].set_ylim(0, 8)
    fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig(f'{OUT}/fig10_출발지_변화.png', facecolor=BG); plt.close(fig); print('fig10')

def fig9():
    """질문 2 가설 A: 성수권 안 성수역 비중(지하철) + 성수동 4개 동별 주말 여가 도착(생활이동)"""
    import s2_living as s2
    sh = O('Q2_성수권안_성수역비중.csv', index_col=0).iloc[:, 0]; sh = sh[[2019, 2023, 2024, 2025, 2026]] if 2019 in sh.index else sh.loc[['2019', '2023', '2024', '2025', '2026']]
    x = s2.prep(); s = x[x.지역.isin(SEONGSU.values()) & (x.구분 == '주말') & (x.목적 == '기타')].groupby(['dt', '지역']).total_cnt.sum().unstack()
    m = s.groupby(s.index.to_period('M')).mean(); yr = m.groupby(m.index.year).mean() / 1000   # 표와 같게: 월평균의 연평균
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.4), dpi=200, gridspec_kw=dict(width_ratios=[1, 1.25])); fig.patch.set_facecolor(BG)
    fig.suptitle('성수 일대는 넓어지기보다 성수역 쪽으로 몰리고 있다', x=0.06, ha='left', fontsize=15, fontweight='bold', color=T1, y=0.97)
    fig.text(0.06, 0.885, '각 해 1~6월 주말, 공휴일 제외 / 왼쪽: 지하철 승하차 / 오른쪽: 서울시·KT 생활이동(도착지, 추정치), 이동 목적 기타(여가 추정)', fontsize=9.5, color=T2)
    for ax in axs:
        ax.set_facecolor(BG); [ax.spines[k].set_visible(False) for k in ['top', 'right']]; [ax.spines[k].set_color(AX) for k in ['left', 'bottom']]
        ax.tick_params(colors=T2); ax.grid(axis='y', color=GR, lw=0.8); ax.set_axisbelow(True)
    a = axs[0]; xs = np.arange(len(sh)); a.bar(xs, sh.values, color=[ '#b8b7b0'] + [PAL[0]] * (len(sh) - 1), width=0.6)
    for x_, v in zip(xs, sh.values): a.text(x_, v + 1, f'{v:.1f}%', ha='center', fontsize=11, color=T1, fontweight='bold')
    a.set_xticks(xs); a.set_xticklabels([str(i) for i in sh.index]); a.set_ylim(0, 75)
    a.set_title('성수·뚝섬·서울숲역 승하차 중 성수역 비중', loc='left', fontsize=12, color=T1)
    b = axs[1]; ys = yr.index.values
    lab = {'성수2가3동': '성수2가3동 (성수역·연무장길)', '성수1가2동': '성수1가2동 (뚝섬역 북쪽)', '성수1가1동': '성수1가1동 (서울숲)', '성수2가1동': '성수2가1동 (남쪽)'}
    cols = {'성수2가3동': PAL[0], '성수1가2동': PAL[2], '성수1가1동': PAL[1], '성수2가1동': '#8a8984'}
    for k, c in cols.items():
        line(b, ys, yr[k], c, lw=2.6 if k == '성수2가3동' else 2, ms=6)
        b.text(ys[-1] + 0.12, yr[k].iloc[-1] + {'성수1가1동': 2.2, '성수2가1동': -2.2}.get(k, 0), f'{lab[k]}\n{yr[k].iloc[0]:.1f}→{yr[k].iloc[-1]:.1f}천 명', va='center', fontsize=9.5, color=T1, fontweight='bold' if k == '성수2가3동' else 'normal')
    b.set_xticks(ys); b.set_xlim(2022.7, 2028.3); b.set_ylim(0, 60); b.set_title('주말 하루 평균 여가(기타) 목적 도착, 천 명', loc='left', fontsize=12, color=T1)
    fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig(f'{OUT}/fig9_성수권_집중.png', facecolor=BG); plt.close(fig); print('fig9')
    return yr

def fig11():
    """질문 2 정점: 같은 달끼리 비교한 성수 주말 점유율 (월 × 연도)"""
    import s1_subway as s1
    x = s1.daily(s1.load()); x = x[(x.구분 == '주말') & (x.연도 >= 2023)]
    m = x.groupby(['연도', '월'])[['성수', '전체']].mean(); sh = (m['성수'] / m['전체'] * 100).unstack('월')
    pg = O('Q2_Page_지하철.csv'); r = pg[(pg.구분 == '주말') & (pg.대상 == '성수')].iloc[0]
    fig, ax = base('같은 달끼리 비교해도 성수는 해마다 올랐고, 오르는 속도도 그대로다',
                   f'성수역 주말 점유율, 월별 / 각 해 1~6월, 공휴일 제외 / Page 추세 검정: 6개 달 모두 2023 < 2024 < 2025 < 2026 (L={r.L:.0f}, p<0.001)\n'
                   f'전년 같은 달 대비 증가율 중앙값 +{r.증가율_23_24:.1f}% → +{r.증가율_24_25:.1f}% → +{r.증가율_25_26:.1f}% (줄어드는 추세 아님, p={r.p_둔화:.2f})')
    ax.set_title('같은 달끼리 비교해도 성수는 해마다 올랐고, 오르는 속도도 그대로다', loc='left', fontsize=15, fontweight='bold', color=T1, pad=46)
    shades = ['#c9dcf3', '#8db7e8', '#4f8fdc', PAL[0]]
    mo = sh.columns.values
    for (y, row), c in zip(sh.iterrows(), shades):
        ax.plot(mo, row.values, color=c, lw=2.8 if y == 2026 else 2.2, marker='o', ms=6, mec='white', mew=1.5, zorder=3)
        ax.text(6.15, row.values[-1], f'{y}', va='center', fontsize=12, color=T1, fontweight='bold' if y == 2026 else 'normal')
    ax.set_xticks(mo); ax.set_xticklabels([f'{k}월' for k in mo]); ax.set_xlim(0.8, 6.6); ax.set_ylim(0.4, 1.0)
    ax.set_ylabel('서울 지하철 전체 대비 승하차 점유율 (%)', color=T2)
    save(fig, 'fig11_성수_같은달비교.png')
    return sh.round(3)

def fig12():
    """Part 2 ①: 주말에 평일보다 더 많이 타는 역 27곳 (2019년 1~6월)"""
    r = O('P2_주말평일비_2019.csv', index_col=0); up = r[r.주말평일비 > 1]
    hi = {'고속터미널', '홍대입구'}
    fig, ax = base(f'주말에 평일보다 더 많이 타는 역은 {len(r)}개 중 {len(up)}개',
                   '2019년 1~6월, 주말 하루 평균 승차 ÷ 평일 하루 평균 승차 (점선 1.0 = 평일과 같음) / 평일 평균 승차 1,000명 이상 역, 평일 공휴일 제외 / 주황: Q5 주말 승차 1·2위 역', figsize=(14, 5.4))
    xs = np.arange(len(up)); cap = 2.2; v = up.주말평일비.clip(upper=cap)
    ax.bar(xs, v, color=[PAL[1] if n in hi else '#9fb4cc' for n in up.index], width=0.7, zorder=2)
    ax.axhline(1, color=T2, ls='--', lw=1.1)
    for i, (n, val) in enumerate(up.주말평일비.items()):
        lab = f'{val:.2f}' if val <= cap else f'{val:.2f} ▲'
        ax.text(i, min(val, cap) + 0.03, lab, ha='center', va='bottom', fontsize=8.5, color=T1, fontweight='bold' if n in hi or val > cap else 'normal', rotation=90)
    ax.set_xticks(xs); ax.set_xticklabels([n.split('(')[0] for n in up.index], rotation=60, ha='right', fontsize=10)
    for t in ax.get_xticklabels():
        if t.get_text() in hi: t.set_color(PAL[1]); t.set_fontweight('bold')
    ax.set_ylim(0, 2.6); ax.set_xlim(-0.7, len(up) - 0.3); ax.set_ylabel('주말 ÷ 평일 (승차)', color=T2)
    save(fig, 'fig12_주말에_더_타는_역.png')
    return up.round(2)

if __name__ == '__main__':
    import sys
    which = sys.argv[1:] or ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    for w in which: print(globals()['fig' + w]())
