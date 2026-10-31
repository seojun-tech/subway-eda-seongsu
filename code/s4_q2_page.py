# 질문 2 보강 (팀원 제안 반영): 같은 달끼리 비교하는 Page 추세 검정
#   - 정점: 성수 점유율이 같은 달끼리 해마다 오르나, 오르는 속도가 줄고 있나
#   - 가설 A: 성수권 안 성수역 비중, 성수동 4개 동의 서울 대비 몫
#   - 가설 B: 성수 ÷ 용산·명동 (누가 더 빨리 크나), 생활이동으로 본 용산 한강로동·명동
import os, zipfile, numpy as np, pandas as pd
from scipy import stats
from common import *
import s1_subway as s1

YRS = [2023, 2024, 2025, 2026]
EXTRA = {'11170625': '한강로동', '11140550': '명동', **SEONGSU}   # 한강로동(용산역·아이파크몰), 명동

def page_up(tab):
    """tab: 행=월(블록), 열=연도 오름차순. 해마다 오른다는 순서로 Page 검정"""
    r = stats.page_trend_test(tab.values); n_up = int((np.diff(tab.values, axis=1) > 0).all(axis=1).sum())
    return r.statistic, r.pvalue, f'{n_up}/{len(tab)}'
def page_down(tab):
    r = stats.page_trend_test(tab.values[:, ::-1]); return r.statistic, r.pvalue

def subway():
    d = s1.load()
    s1.GROUPS.update({'용산+신용산': [('경부선', '용산'), ('4호선', '신용산')], '명동+을지로입구': [('4호선', '명동'), ('2호선', '을지로입구')]})
    x = s1.daily(d); nd = x[(x.구분 != '제외') & x.연도.isin(YRS)].copy()
    nd['성수역비중'] = nd['성수'] / nd['성수권'] * 100
    G = ['성수', '용산', '명동', '뚝섬', '서울숲', '건대입구', '용산+신용산', '명동+을지로입구']
    m = nd.groupby(['구분', '월', '연도'])[G + ['전체']].mean()
    for g in G: m[g + '_sh'] = m[g] / m['전체'] * 100
    rows = []
    def tab(t, col): return m.loc[t][col].unstack('연도')[YRS]
    for t in ['주말', '평일']:
        for g in ['성수', '용산', '명동', '뚝섬', '서울숲', '건대입구']:
            tb = tab(t, g + '_sh'); L, p, n = page_up(tb); _, pd_ = page_down(tb)
            gr = (tb.pct_change(axis=1).iloc[:, 1:] * 100); med = gr.median()
            _, p_slow = page_down(gr)    # 증가율이 해마다 작아지는 순서(둔화)
            rows.append(dict(구분=t, 대상=g, 지표='점유율', L=L, p_상승=p, p_하락=pd_, 상승달=n,
                             **{f'{y}': round(tb[y].mean(), 3) for y in YRS},
                             증가율_23_24=round(med.iloc[0], 1), 증가율_24_25=round(med.iloc[1], 1), 증가율_25_26=round(med.iloc[2], 1), p_둔화=p_slow))
    w = nd[nd.구분 == '주말']
    tb = w.groupby(['월', '연도'])['성수역비중'].mean().unstack()[YRS]; L, p, n = page_up(tb)
    rows.append(dict(구분='주말', 대상='성수권 안 성수역 비중', 지표='비중', L=L, p_상승=p, 상승달=n, **{f'{y}': round(tb[y].mean(), 1) for y in YRS}))
    for b in ['용산', '명동', '용산+신용산', '명동+을지로입구']:
        mm = m.loc['주말']; r = (mm['성수'] / mm[b]).unstack('연도')[YRS]; L, p, n = page_up(r)
        yr = w.groupby('연도')[['성수', b]].mean(); ry = yr['성수'] / yr[b]
        rows.append(dict(구분='주말', 대상=f'성수÷{b}', 지표='상대 점유율', L=L, p_상승=p, 상승달=n, **{f'{y}': round(ry[y], 2) for y in YRS}))
    out = pd.DataFrame(rows); out.to_csv(f'{OUT}/Q2_Page_지하철.csv', index=False, encoding='utf-8-sig')
    for k in ['용산+신용산', '명동+을지로입구']: s1.GROUPS.pop(k)
    return out

def extra_dongs():
    """생활이동 도착지 원본에서 한강로동·명동·성수동 4개 동·서울 전체의 날짜×목적 도착 합계"""
    c = f'{CACHE}/lm_extra_dong.pkl'
    if os.path.exists(c): return pd.read_pickle(c)
    parts = []
    for y in YRS:
        for mo in range(1, 7):
            with zipfile.ZipFile(f'{LM_DIR}/seoul_purpose_admdong1_in_{y}{mo:02d}.zip') as f:
                for n in f.namelist():
                    d = pd.read_csv(f.open(n), dtype={'d_admdong_cd': str}, usecols=['d_admdong_cd', 'move_purpose', 'total_cnt', 'etl_ymd'])
                    d = d[d.d_admdong_cd.str.startswith('11')]; d['지역'] = d.d_admdong_cd.map(EXTRA).fillna('그외')
                    parts.append(d.groupby(['etl_ymd', '지역', 'move_purpose']).total_cnt.sum().reset_index())
    x = pd.concat(parts); x.to_pickle(c); return x

def living():
    x = extra_dongs().copy(); x['dt'] = pd.to_datetime(x.etl_ymd.astype(str)); hol = holiday_set(YRS)
    wk = {t: day_type(t, hol) for t in x.dt.unique()}; x = x[(x.dt.map(wk) == '주말') & (x.move_purpose == 7)]           # 주말, 기타(여가 추정)
    p = x.pivot_table(index='dt', columns='지역', values='total_cnt', aggfunc='sum')
    p['서울 전체'] = p.sum(axis=1); p['성수동'] = p[list(SEONGSU.values())].sum(axis=1)
    sh = p.div(p['서울 전체'], axis=0) * 100; mm = sh.groupby([sh.index.month, sh.index.year]).mean()
    yr = p.groupby(p.index.year).mean(); rows = []
    for c in ['성수2가3동', '성수1가2동', '성수1가1동', '성수2가1동', '성수동', '한강로동', '명동']:
        tb = mm[c].unstack()[YRS]; L, pu, n = page_up(tb); _, pdn = page_down(tb); y = tb.mean()
        rows.append(dict(대상=c, 인원2023=round(yr.loc[2023, c]), 인원2026=round(yr.loc[2026, c]), 인원변화=round((yr.loc[2026, c] / yr.loc[2023, c] - 1) * 100, 1),
                         몫2023=round(y[2023], 3), 몫2026=round(y[2026], 3), 몫변화=round((y[2026] / y[2023] - 1) * 100, 1), p_상승=pu, p_하락=pdn, 상승달=n,
                         몫2026lt2025=int((tb[2026] < tb[2025]).sum())))
    for a in ['성수2가3동', '성수동']:
        r = (mm[a] / mm['한강로동']).unstack()[YRS]; _, pu, _ = page_up(r); _, pdn = page_down(r)
        rows.append(dict(대상=f'{a}÷한강로동', 몫2023=round(r[2023].mean(), 2), 몫2026=round(r[2026].mean(), 2), p_상승=pu, p_하락=pdn))
    # 목적 코드 구성 (한강로동·명동은 성수와 같은 잣대인지 확인용)
    allp = extra_dongs().groupby(['지역', 'move_purpose']).total_cnt.sum().unstack().fillna(0)
    (allp.div(allp.sum(axis=1), axis=0) * 100).round(2).to_csv(f'{OUT}/Q2_목적구성_동별.csv', encoding='utf-8-sig')
    out = pd.DataFrame(rows); out.to_csv(f'{OUT}/Q2_Page_생활이동.csv', index=False, encoding='utf-8-sig'); return out

def part2_weekend():
    """Part 2 ①: 2019년 1~6월 역별 주말 하루 평균 승차 ÷ 평일 하루 평균 승차 (팀 단계 기준: 평일 공휴일만 제외)"""
    import holidays
    d = s1.load(); d = d[d.연도 == 2019]
    hs = {pd.Timestamp(v) for v in holidays.KR(years=2019)}
    d = d[~(d.사용일자.isin(hs) & (d.요일 < 5))]
    g = d.groupby(['역명', '사용일자'])['승차총승객수'].sum().reset_index(); g['주말'] = g.사용일자.dt.dayofweek >= 5
    m = g.groupby(['역명', '주말'])['승차총승객수'].mean().unstack()
    m = m[(m[False] >= 1000) & (m.index != '양평')]
    r = (m[True] / m[False]).rename('주말평일비').to_frame(); r['평일'] = m[False].round(); r['주말'] = m[True].round()
    r = r.sort_values('주말평일비', ascending=False); r['순위'] = range(1, len(r) + 1)
    r.to_csv(f'{OUT}/P2_주말평일비_2019.csv', encoding='utf-8-sig'); return r

def run():
    a = subway(); b = living(); c = part2_weekend()
    print(a.round(4).to_string()); print(b.round(4).to_string())
    print('역 수', len(c), '주말>평일', int((c.주말평일비 > 1).sum()), '중앙값', round(c.주말평일비.median(), 2)); print(c.head(20).round(2).to_string())
    return a, b, c

if __name__ == '__main__': run()
