# 3단계: 생활이동 출발-도착지(OA-22299) 36일 (2023·2026 각 1~6월, 달마다 수·토·일 1일씩): 질문 1 "어디서 데려왔나"
# 같은 동네 안 이동 제외, 서울 전체(서울 도착 이동 전부)를 같은 날짜 기준선으로 계산
import glob, os, zipfile, numpy as np, pandas as pd
from scipy import stats
from common import *

S = list(SEONGSU); H = list(HWAYANG)
G2 = ['11200670', '11200690']   # 성수2가1동·2가3동 (성수역 쪽)
S1 = ['11200650']               # 성수1가1동 (서울숲)
WD = {2: '수', 5: '토', 6: '일'}

def wmed(v, w):
    o = np.argsort(v); c = np.cumsum(w[o]) / w.sum(); return v[o][np.searchsorted(c, 0.5)]

def region(o):
    if o.startswith('11200') or o.startswith('11215'): return '성동광진'
    if o.startswith('11'): return '서울다른구'
    return '서울밖'

def ingest():
    """하루 파일마다: 성수·화양 관련 이동만 남긴 표(cache/od_YYYYMMDD.pkl) + 서울 전체 기준선 한 줄"""
    bc = f'{CACHE}/od_baseline.csv'
    base = pd.read_csv(bc, dtype={'날짜': str}) if os.path.exists(bc) else pd.DataFrame(columns=['날짜'])
    rows = base.to_dict('records'); done = set(base.날짜)
    for z in sorted(glob.glob(f'{OD_DIR}/seoul_purpose_admdong4_in_202[36]*.zip')):
        day = z[-12:-4]; kp = f'{CACHE}/od_{day}.pkl'
        if day in done and os.path.exists(kp): continue
        with zipfile.ZipFile(z) as f:
            d = pd.read_csv(f.open(f.namelist()[0]), dtype={'o_admdong_cd': str, 'd_admdong_cd': str, 'st_time_cd': str, 'fns_time_cd': str})
        assert str(d.etl_ymd.iloc[0]) == day and d.etl_ymd.nunique() == 1
        d[d.d_admdong_cd.isin(S + H) | d.o_admdong_cd.isin(S + H)].to_pickle(kp)
        b = d[d.d_admdong_cd.str.startswith('11') & (d.o_admdong_cd != d.d_admdong_cd)]
        w = b.total_cnt.values; tot = w.sum()
        rows.append(dict(날짜=day, 합계=tot, 중앙거리km=wmed(b.move_dist.values, w) / 1000,
                         서울밖=w[~b.o_admdong_cd.str.startswith('11').values].sum() / tot * 100,
                         이삼십대=b['2030_cnt'].sum() / tot * 100))
        print(day, flush=True)
    base = pd.DataFrame(rows); base.to_csv(bc, index=False, encoding='utf-8-sig'); return base

def metrics(base):
    rows = []; gu = {}
    for kp in sorted(glob.glob(f'{CACHE}/od_202[36]*.pkl')):
        day = kp[-12:-4]; d = pd.read_pickle(kp); wd = WD[pd.Timestamp(day).dayofweek]
        fl = lambda o, dd: d[d.o_admdong_cd.isin(o) & d.d_admdong_cd.isin(dd)].total_cnt.sum()
        for name, dest, other in [('성수동', S, H), ('화양동', H, S)]:
            x = d[d.d_admdong_cd.isin(dest) & ~d.o_admdong_cd.isin(dest)]; w = x.total_cnt.values; tot = w.sum()
            r = x.o_admdong_cd.map(region).values
            out = d[d.o_admdong_cd.isin(dest) & ~d.d_admdong_cd.isin(dest)].total_cnt.sum()
            row = dict(날짜=day, 연=day[:4], 월=day[4:6], 요일=wd, 도착=name, 외부유입=tot, 중앙거리km=wmed(x.move_dist.values, w) / 1000,
                       성동광진=w[r == '성동광진'].sum() / tot * 100, 서울다른구=w[r == '서울다른구'].sum() / tot * 100, 서울밖=w[r == '서울밖'].sum() / tot * 100,
                       이삼십대=x['2030_cnt'].sum() / tot * 100,
                       상대에서온비율=fl(other, dest) / tot * 100,      # 성수: 화양→성수 / 성수 외부유입
                       상대로간비율=fl(dest, other) / out * 100)        # 성수: 성수→화양(건대) / 성수에서 나간 이동
            if name == '성수동':
                row['성수2가_서울숲비율'] = fl(G2, S1) / d[d.o_admdong_cd.isin(G2) & ~d.d_admdong_cd.isin(G2)].total_cnt.sum() * 100
            rows.append(row); gu[(name, day[:4])] = gu.get((name, day[:4]), []) + [x.groupby(x.o_admdong_cd.str[:5]).total_cnt.sum()]
    t = pd.DataFrame(rows)
    b = base[['날짜', '합계', '중앙거리km', '서울밖', '이삼십대']].rename(columns=lambda c: c if c == '날짜' else c + '_서울')
    t = t.merge(b, on='날짜')
    t['서울몫'] = t.외부유입 / t.합계_서울 * 100; t['거리배수'] = t.중앙거리km / t.중앙거리km_서울
    t['서울밖차'] = t.서울밖 - t.서울밖_서울; t['이삼십대차'] = t.이삼십대 - t.이삼십대_서울
    return t, gu

def wil(a, b):
    return stats.wilcoxon(a, b).pvalue if len(a) > 0 and (np.asarray(a) != np.asarray(b)).any() else np.nan

GROUPS = {'수': ['수'], '토': ['토'], '일': ['일'], '주말(토+일)': ['토', '일']}

def tests(t):
    res = []
    for g, ws in GROUPS.items():   # 2026 성수 vs 화양, 같은 날짜
        a = t[(t.연 == '2026') & (t.도착 == '성수동') & t.요일.isin(ws)].sort_values('날짜'); c = t[(t.연 == '2026') & (t.도착 == '화양동') & t.요일.isin(ws)].sort_values('날짜')
        for k in ['중앙거리km', '성동광진', '서울다른구', '서울밖', '이삼십대', '상대에서온비율']:
            res.append(dict(비교='2026 성수vs화양', 요일=g, 지표=k, A=a[k].mean(), B=c[k].mean(), 방향=f'{int((a[k].values > c[k].values).sum())}/{len(a)} 성수가 큼', p=wil(a[k].values, c[k].values)))
    for name in ['성수동', '화양동']:   # 2023 vs 2026, 같은 달·같은 요일 짝
        for g, ws in GROUPS.items():
            a = t[(t.연 == '2023') & (t.도착 == name) & t.요일.isin(ws)].sort_values(['요일', '월']); c = t[(t.연 == '2026') & (t.도착 == name) & t.요일.isin(ws)].sort_values(['요일', '월'])
            for k in ['외부유입', '서울몫', '거리배수', '서울밖차', '이삼십대차', '상대에서온비율', '상대로간비율', '성수2가_서울숲비율']:
                if k not in a or a[k].isna().all(): continue
                res.append(dict(비교=f'{name} 2023→2026', 요일=g, 지표=k, A=a[k].mean(), B=c[k].mean(), 방향=f'{int((c[k].values > a[k].values).sum())}/{len(a)} 늘어남', p=wil(c[k].values, a[k].values)))
    return pd.DataFrame(res)

def gu_change(gu):
    """출발 구별 변화: 성수로 온 이동과 화양으로 온 이동이 같은 구에서 반대로 움직였나 (빼앗기 확인). 두 해 모두 있는 구만"""
    agg = lambda n, y: pd.concat(gu[(n, y)], axis=1).fillna(0).mean(axis=1)
    D = pd.DataFrame({'성수23': agg('성수동', '2023'), '성수26': agg('성수동', '2026'), '화양23': agg('화양동', '2023'), '화양26': agg('화양동', '2026')}).fillna(0)
    big = D[(D.성수23 >= 500) & (D.화양23 >= 200) & (D.성수26 > 0) & (D.화양26 > 0)].copy()
    big['성수변화'] = np.log(big.성수26 / big.성수23); big['화양변화'] = np.log(big.화양26 / big.화양23)
    r = stats.spearmanr(big.성수변화, big.화양변화); b2 = big.drop(['11215', '11200'], errors='ignore'); r2 = stats.spearmanr(b2.성수변화, b2.화양변화)
    return big, dict(구수=len(big), rho=r.statistic, p=r.pvalue, 광진성동제외_구수=len(b2), 광진성동제외_rho=r2.statistic, 광진성동제외_p=r2.pvalue)

def representativeness(t):
    """뽑은 날이 같은 해 1~6월 같은 요일 전체와 다르지 않은가: 도착지 데이터(매일 있음)로 확인 + 몫 변화를 전체 날짜로 다시"""
    import s2_living
    x = s2_living.prep(); x = x[x.지역.isin(['성수동', '화양동', '서울 전체']) & x.dt.dt.year.isin([2023, 2026])]
    tot = x.groupby(['dt', '지역']).total_cnt.sum().unstack()
    pur = x.pivot_table(index='dt', columns=['지역', '목적'], values='total_cnt', aggfunc='sum')
    day = pd.DataFrame({'성수몫': tot.성수동 / tot['서울 전체'] * 100, '화양몫': tot.화양동 / tot['서울 전체'] * 100,
                        '성수출근': pur[('성수동', '출근')] / tot.성수동 * 100, '성수기타': pur[('성수동', '기타')] / tot.성수동 * 100})
    day['연'] = day.index.year; day['요일'] = day.index.dayofweek.map({2: '수', 5: '토', 6: '일'})
    day['뽑음'] = day.index.isin(pd.to_datetime(t.날짜.unique()))
    rows = []
    for (y, w), g in day.dropna(subset=['요일']).groupby(['연', '요일']):
        for k in ['성수몫', '화양몫', '성수출근', '성수기타']:
            p = g[g.뽑음][k]
            rows.append(dict(연=y, 요일=w, 지표=k, 뽑은날수=len(p), 뽑은날평균=p.mean(), 전체평균=g[k].mean(), 전체일수=len(g),
                             p_뽑은날vs나머지=stats.mannwhitneyu(p, g[~g.뽑음][k]).pvalue))
    rep = pd.DataFrame(rows); full = []
    for w in ['수', '토', '일']:
        for k in ['성수몫', '화양몫']:
            a = day[(day.연 == 2023) & (day.요일 == w)][k]; b = day[(day.연 == 2026) & (day.요일 == w)][k]
            full.append(dict(요일=w, 지표=k, 평균2023=a.mean(), 평균2026=b.mean(), 일수2023=len(a), 일수2026=len(b), p=stats.mannwhitneyu(b, a).pvalue))
    return rep, pd.DataFrame(full)

def run():
    base = ingest(); base.to_csv(f'{OUT}/Q1_출발지_서울전체기준.csv', index=False, encoding='utf-8-sig')
    t, gu = metrics(base); t.to_csv(f'{OUT}/Q1_출발지_36일_지표.csv', index=False, encoding='utf-8-sig')
    R = tests(t); R.to_csv(f'{OUT}/Q1_출발지_36일_검정.csv', index=False, encoding='utf-8-sig')
    big, gc = gu_change(gu); big.to_csv(f'{OUT}/Q1_출발구_변화.csv', encoding='utf-8-sig'); pd.DataFrame([gc]).to_csv(f'{OUT}/Q1_출발구_변화_상관.csv', index=False, encoding='utf-8-sig')
    rep, full = representativeness(t)
    rep.to_csv(f'{OUT}/Q1_출발지_표본대표성.csv', index=False, encoding='utf-8-sig'); full.to_csv(f'{OUT}/Q1_도착몫_전체날짜_2023vs2026.csv', index=False, encoding='utf-8-sig')
    # 서울 전체 기준선 자체의 변화 (2023→2026)
    base['연'] = base.날짜.str[:4]; bl = base.groupby('연')[['중앙거리km', '서울밖', '이삼십대']].median()
    return dict(t=t, 검정=R, 구별=gc, 대표성=rep, 전체날짜=full, 기준선=bl)

if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
    R = run()
    for k in ['검정', '구별', '대표성', '전체날짜', '기준선']:
        v = R[k]; print('==', k); print(v.round(4).to_string() if isinstance(v, pd.DataFrame) else v)
