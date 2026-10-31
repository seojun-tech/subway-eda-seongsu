# 2단계: 생활이동 도착지(이동 목적) 2023~2026 각 1~6월: 질문 3, 가설 C(학생 수요), 질문 2 가설 A(성수동 동별)
import glob, os, zipfile, numpy as np, pandas as pd
from scipy import stats
from common import *

P = {1: '출근', 2: '등교', 3: '귀가', 4: '쇼핑·관광', 5: '쇼핑·관광', 6: '병원', 7: '기타'}   # 4·5는 성수·화양에 없음 → 기타를 여가로 추정
AGE = ['male_20_cnt', 'male_30_cnt', 'feml_20_cnt', 'feml_30_cnt']
MONTHS = [f'{y}{m:02d}' for y in range(2023, 2027) for m in range(1, 7)]

def compact():
    """원본 zip을 날짜×지역(성수동 4개 동·화양동·서울 전체)×시각×목적으로 줄여 저장"""
    c = f'{CACHE}/lm_compact.pkl'
    if os.path.exists(c): return pd.read_pickle(c)
    parts = []
    for m in MONTHS:
        z = f'{LM_DIR}/seoul_purpose_admdong1_in_{m}.zip'
        with zipfile.ZipFile(z) as f:
            for n in f.namelist():
                d = pd.read_csv(f.open(n), dtype={'d_admdong_cd': str, 'time_cd': str},
                                usecols=['d_admdong_cd', 'time_cd', 'move_purpose', 'total_cnt', 'etl_ymd'] + AGE)
                d = d[d.d_admdong_cd.str.startswith('11')]
                d['h'] = d.time_cd.str[:2].astype(int); d['a2030'] = d[AGE].sum(axis=1)
                d['지역'] = d.d_admdong_cd.map({**SEONGSU, **HWAYANG}).fillna('그외')
                a = d.groupby(['etl_ymd', '지역', 'h', 'move_purpose'], as_index=False)[['total_cnt', 'a2030']].sum()
                s = a.groupby(['etl_ymd', 'h', 'move_purpose'], as_index=False)[['total_cnt', 'a2030']].sum().assign(지역='서울 전체')
                parts.append(pd.concat([a[a.지역 != '그외'], s]))
        print(m)
    x = pd.concat(parts, ignore_index=True); x.to_pickle(c); return x

def prep():
    x = compact().copy()
    x['dt'] = pd.to_datetime(x.etl_ymd.astype(str)); hol = holiday_set(range(2023, 2027))
    x['구분'] = [day_type(t, hol) for t in x.dt]; x['목적'] = x.move_purpose.map(P)
    x = x[x.구분 != '제외']
    s = x[x.지역.str.startswith('성수')].groupby(['etl_ymd', 'dt', '구분', 'h', 'move_purpose', '목적'], as_index=False)[['total_cnt', 'a2030']].sum().assign(지역='성수동')
    return pd.concat([x, s], ignore_index=True)

M = {'평일 출근 비중': ('평일', lambda g: g[g.목적 == '출근'].total_cnt.sum() / g.total_cnt.sum() * 100),
     '주말 기타 비중': ('주말', lambda g: g[g.목적 == '기타'].total_cnt.sum() / g.total_cnt.sum() * 100),
     '주말 10~16시 비중': ('주말', lambda g: g[(g.h >= 10) & (g.h <= 16)].total_cnt.sum() / g.total_cnt.sum() * 100),
     '주말 18~23시 비중': ('주말', lambda g: g[(g.h >= 18) & (g.h <= 23)].total_cnt.sum() / g.total_cnt.sum() * 100),
     '주말 20~30대 비중': ('주말', lambda g: g.a2030.sum() / g.total_cnt.sum() * 100)}
REG = ['성수동', '화양동', '서울 전체']

def daily_metric(x, name):
    t, fn = M[name]; s = x[(x.구분 == t) & x.지역.isin(REG)]
    return s.groupby(['dt', '지역']).apply(fn, include_groups=False).unstack('지역')

def run():
    x = prep(); R = {}
    y26 = x[x.dt.dt.year == 2026]
    # 질문 3 (2026 1~6월): 합계 비중 + 같은 날짜끼리 Wilcoxon, 달마다 방향
    rows = []
    for name, (t, fn) in M.items():
        s = y26[(y26.구분 == t) & y26.지역.isin(REG)]
        pooled = {r: fn(s[s.지역 == r]) for r in REG}; dl = daily_metric(y26, name)
        mon = dl.groupby(dl.index.month).mean(); same = int(((mon.성수동 - mon.화양동) * (pooled['성수동'] - pooled['화양동']) > 0).sum())
        rows.append(dict(지표=name, 일수=len(dl), **{r: round(v, 1) for r, v in pooled.items()},
                         성수가화양보다큰날=f'{int((dl.성수동 > dl.화양동).sum())}/{len(dl)}', p_성수vs화양=stats.wilcoxon(dl.성수동, dl.화양동).pvalue,
                         성수가서울보다큰날=f'{int((dl.성수동 > dl["서울 전체"]).sum())}/{len(dl)}', p_성수vs서울=stats.wilcoxon(dl.성수동, dl['서울 전체']).pvalue,
                         달별같은방향=f'{same}/6'))
    q3 = pd.DataFrame(rows); q3.to_csv(f'{OUT}/Q3_생활이동_2026상반기.csv', index=False, encoding='utf-8-sig'); R['Q3'] = q3
    w = y26[(y26.구분 == '주말') & y26.지역.isin(REG)].groupby(['지역', 'h']).total_cnt.sum()
    prof = (w / w.groupby('지역').transform('sum') * 100).unstack('지역'); prof.to_csv(f'{OUT}/Q3_주말_시간대별_도착비중_2026.csv', encoding='utf-8-sig')
    R['시간대'] = prof; R['피크'] = prof.idxmax().to_dict()

    # 질문 3 변화 (2023~2026): 해마다 비중 + "성수 − 서울" 일별 차이와 연도 Spearman
    rows = []; tr = []
    for name in ['평일 출근 비중', '주말 기타 비중', '주말 10~16시 비중']:
        t, fn = M[name]
        for y in range(2023, 2027):
            s = x[(x.dt.dt.year == y) & (x.구분 == t)]
            rows.append(dict(지표=name, 연도=y, **{r: round(fn(s[s.지역 == r]), 1) for r in REG}))
        dl = daily_metric(x, name); diff = dl.성수동 - dl['서울 전체']; r = stats.spearmanr(diff.index.year, diff.values)
        dy = diff.groupby(diff.index.year).median()   # 해마다 일별 차이의 중앙값
        tr.append(dict(지표=name, n=len(diff), **{f'차이{y}_중앙값': round(dy[y], 2) for y in range(2023, 2027)}, rho=round(r.statistic, 2), p=r.pvalue))
    R['Q3변화'] = pd.DataFrame(rows); R['Q3변화'].to_csv(f'{OUT}/Q3_생활이동_상반기_2023_2026.csv', index=False, encoding='utf-8-sig')
    R['Q3차이추세'] = pd.DataFrame(tr); R['Q3차이추세'].to_csv(f'{OUT}/Q3_생활이동_성수와차이_추세.csv', index=False, encoding='utf-8-sig')

    # 가설 C: 화양동 평일 등교: 방학(1~2월) vs 학기(3~6월), 학기 등교 인원 연도 추세
    rows = []
    for r_ in REG:
        s = x[(x.지역 == r_) & (x.구분 == '평일')]
        dd = s.groupby('dt').apply(lambda g: pd.Series({'등교': g[g.목적 == '등교'].total_cnt.sum(), '전체': g.total_cnt.sum()}), include_groups=False)
        dd['비중'] = dd.등교 / dd.전체 * 100
        for y in range(2023, 2027):
            z = dd[dd.index.year == y]; a = z[z.index.month <= 2]; b = z[z.index.month >= 3]
            rows.append(dict(지역=r_, 연도=y, 방학_등교=round(a.등교.mean()), 학기_등교=round(b.등교.mean()), 학기배수=round(b.등교.mean() / a.등교.mean(), 2),
                             방학_비중=round(a.비중.mean(), 1), 학기_비중=round(b.비중.mean(), 1), p_비중=stats.mannwhitneyu(b.비중, a.비중).pvalue))
        if r_ == '화양동':
            sem = dd[dd.index.month >= 3]; rr = stats.spearmanr(sem.index.year, sem.등교); R['화양학기등교추세'] = dict(rho=rr.statistic, p=rr.pvalue)
    R['가설C'] = pd.DataFrame(rows); R['가설C'].to_csv(f'{OUT}/가설C_화양동_등교.csv', index=False, encoding='utf-8-sig')

    # 질문 2 가설 A: 성수동 4개 동별 주말 기타(여가 추정) 도착: 월평균 하루 인원과 성수동 안 비중, 월별 24개 Spearman
    rows = []
    for t in ['주말', '평일']:
        s = x[x.지역.isin(SEONGSU.values()) & (x.구분 == t) & (x.목적 == '기타')].groupby(['dt', '지역']).total_cnt.sum().unstack()
        m = s.groupby(s.index.to_period('M')).mean(); sh = m.div(m.sum(axis=1), axis=0) * 100; yr = pd.Series(sh.index.year, index=sh.index)
        for c in sh.columns:
            r = stats.spearmanr(np.arange(len(sh)), sh[c])
            rows.append(dict(구분=t + ' 기타', 동=c, 인원2023=round(m[c][yr == 2023].mean()), 인원2026=round(m[c][yr == 2026].mean()),
                             비중2023=round(sh[c][yr == 2023].mean(), 1), 비중2026=round(sh[c][yr == 2026].mean(), 1), rho=round(r.statistic, 2), p=r.pvalue))
    R['동별'] = pd.DataFrame(rows); R['동별'].to_csv(f'{OUT}/Q2_성수동_동별_여가도착.csv', index=False, encoding='utf-8-sig')
    R['x'] = x
    return R

if __name__ == '__main__':
    R = run()
    for k, v in R.items():
        if k not in ('x', '시간대'): print('==', k); print(v if not isinstance(v, pd.DataFrame) else v.round(3).to_string())
