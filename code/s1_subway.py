# 1단계: 지하철 승하차 (2019~2026 각 1~6월): 질문 1(2019 대비), 질문 2(추세·성수권 안 비중), 금요일, 주말÷평일 순위
import glob, os, re, numpy as np, pandas as pd
from scipy import stats
from common import *

RENAME = {'낙성대': '낙성대(강감찬)', '동대문역사문화공원': '동대문역사문화공원(DDP)', '당고개': '불암산', '삼각지': '삼각지(전쟁기념관)',
          '뚝섬유원지': '자양(뚝섬한강공원)', '상봉(시외버스터미널)': '상봉', '용마산': '용마산(용마폭포공원)', '마곡나루': '마곡나루(서울식물원)', '지제': '평택지제'}
GROUPS = {'성수': [('2호선', '성수')], '뚝섬': [('2호선', '뚝섬')], '서울숲': [('분당선', '서울숲')],
          '건대입구': [('2호선', '건대입구'), ('7호선', '건대입구')], '용산': [('경부선', '용산')], '명동': [('4호선', '명동')],
          '홍대입구': [('2호선', '홍대입구'), ('경의선', '홍대입구'), ('공항철도 1호선', '홍대입구')],
          '고속터미널': [('3호선', '고속터미널'), ('7호선', '고속터미널'), ('9호선', '고속터미널')]}
GROUPS['성수권'] = GROUPS['성수'] + GROUPS['뚝섬'] + GROUPS['서울숲']
YEARS = range(2019, 2027)

def read(f):
    enc = 'utf-8-sig' if open(f, 'rb').read(3) == b'\xef\xbb\xbf' else 'cp949'
    d = pd.read_csv(f, encoding=enc, index_col=False, usecols=range(6), low_memory=False)
    d['사용일자'] = pd.to_datetime(d['사용일자'].astype(str), format='%Y%m%d')
    return d[d['사용일자'].dt.month <= 6]                         # 모든 해 1~6월만

def load():
    c = f'{CACHE}/subway_h1.pkl'
    if os.path.exists(c): return pd.read_pickle(c)
    fs = sorted(glob.glob(f'{SUBWAY_DIR}/CARD_SUBWAY_MONTH_20*.csv'))
    d = pd.concat([read(f) for f in fs], ignore_index=True)
    for k in ['승차총승객수', '하차총승객수']: d[k] = d[k].astype(float)
    d['역명'] = d['역명'].replace(RENAME)
    d = d.groupby(['사용일자', '노선명', '역명'], as_index=False)[['승차총승객수', '하차총승객수']].sum()
    d['연도'] = d['사용일자'].dt.year; d['승하차'] = d['승차총승객수'] + d['하차총승객수']
    n = d.groupby(['노선명', '역명'])['연도'].nunique(); common = set(n[n == len(YEARS)].index)
    d['공통'] = [k in common for k in zip(d['노선명'], d['역명'])]
    hol = holiday_set(YEARS)
    d['요일'] = d['사용일자'].dt.dayofweek                        # 0=월, 사용일자에서 새로 계산
    d['구분'] = [day_type(t, hol) for t in d['사용일자']]
    d.to_pickle(c); return d

def daily(d):
    """날짜별 역 그룹 승하차와 공통 587개 역 합계(분모)"""
    out = [d[d['공통']].groupby('사용일자')['승하차'].sum().rename('전체')]
    key = pd.Series(list(zip(d['노선명'], d['역명'])))
    for g, ks in GROUPS.items():
        out.append(d[key.isin(ks).values].groupby('사용일자')['승하차'].sum().rename(g))
    x = pd.concat(out, axis=1).reset_index()
    x['연도'] = x.사용일자.dt.year; x['월'] = x.사용일자.dt.month; x['요일'] = x.사용일자.dt.dayofweek
    hol = holiday_set(YEARS); x['구분'] = [day_type(t, hol) for t in x.사용일자]
    for g in GROUPS: x[g + '_sh'] = x[g] / x['전체'] * 100
    return x

def run():
    d = load(); x = daily(d); R = {}
    print('공통 역 수', d[d.공통].groupby(['노선명', '역명']).ngroups)
    nd = x[x.구분 != '제외']

    # 성수 점유율 (해·구분별 하루 평균 승하차의 비)
    import holidays as _h                                          # 공휴일 점유율: 그해 법정 공휴일 전부(주말에 걸린 날 포함, 2026 노동절 포함)
    hs = {pd.Timestamp(v) for y in YEARS for v in _h.KR(years=y)}
    hol_wd = x[x.사용일자.isin(hs)]
    rows = []
    for y in YEARS:
        r = {'연도': y}
        for t in ['평일', '주말']:
            s = nd[(nd.연도 == y) & (nd.구분 == t)]; r[t] = s['성수'].mean() / s['전체'].mean() * 100
        s = hol_wd[hol_wd.연도 == y]; r['공휴일'] = s['성수'].mean() / s['전체'].mean() * 100
        rows.append(r)
    sh = pd.DataFrame(rows).round(3); sh.to_csv(f'{OUT}/성수_점유율_평일주말공휴일.csv', index=False, encoding='utf-8-sig'); R['성수점유율'] = sh

    # 질문 1: 2019 vs 2023~2026 각 해, 하루 점유율 Mann-Whitney + Holm(네 해)
    rows = []
    for g in ['성수', '뚝섬', '서울숲', '건대입구', '성수권']:
        for t in ['평일', '주말']:
            a = nd[(nd.연도 == 2019) & (nd.구분 == t)][g + '_sh']; ps = []; tmp = []
            for y in [2023, 2024, 2025, 2026]:
                b = nd[(nd.연도 == y) & (nd.구분 == t)][g + '_sh']; u = stats.mannwhitneyu(a, b)
                ps.append(u.pvalue); tmp.append(dict(역=g, 구분=t, 비교연도=y, n_2019=len(a), n_비교=len(b), 중앙값_2019=round(a.median(), 3),
                                                     중앙값_비교=round(b.median(), 3), 배수=round(b.median() / a.median(), 2), U=u.statistic, p=u.pvalue))
            for r, p in zip(tmp, holm(ps)): r['p_Holm'] = p; r['판정'] = '증가' if r['배수'] > 1 else '감소'; rows.append(r)
    q1 = pd.DataFrame(rows); q1.to_csv(f'{OUT}/Q1_2019대비_MannWhitney.csv', index=False, encoding='utf-8-sig'); R['Q1'] = q1
    pw = []
    for y in YEARS:
        a = nd[(nd.연도 == y) & (nd.구분 == '평일')]['성수_sh']; b = nd[(nd.연도 == y) & (nd.구분 == '주말')]['성수_sh']
        pw.append(dict(연도=y, 평일중앙=a.median(), 주말중앙=b.median(), p=stats.mannwhitneyu(a, b).pvalue))
    R['평일vs주말'] = pd.DataFrame(pw); R['평일vs주말'].to_csv(f'{OUT}/Q1_성수_평일vs주말.csv', index=False, encoding='utf-8-sig')

    # 2019 대비 주말 점유율 배수 (해별 하루 평균의 비): 그림 2
    idx = {}
    for g in ['성수', '뚝섬', '서울숲', '건대입구']:
        v = {y: nd[(nd.연도 == y) & (nd.구분 == '주말')][g].mean() / nd[(nd.연도 == y) & (nd.구분 == '주말')]['전체'].mean() for y in YEARS}
        idx[g] = {y: v[y] / v[2019] * 100 for y in YEARS}
    R['주말지수'] = pd.DataFrame(idx); R['주말지수'].round(1).to_csv(f'{OUT}/Q1_주말점유율_2019지수.csv', encoding='utf-8-sig')
    wk = {}
    for g in ['성수', '뚝섬', '서울숲', '건대입구']:
        v = {y: nd[(nd.연도 == y) & (nd.구분 == '평일')][g].mean() / nd[(nd.연도 == y) & (nd.구분 == '평일')]['전체'].mean() for y in YEARS}
        wk[g] = v[2026] / v[2019]
    R['평일배수2026'] = wk

    # 질문 2: 2023~2026 같은 1~6월 월별 점유율 24개와 연도 Spearman (+ 2024 시작 확인)
    G = list(GROUPS)
    m = nd[nd.연도 >= 2023].groupby(['연도', '월', '구분'])[G + ['전체']].mean()
    for g in G: m[g] = m[g] / m['전체'] * 100
    m = m.reset_index(); m.to_csv(f'{OUT}/월별_점유율.csv', index=False, encoding='utf-8-sig')
    rows = []
    for start in [2023, 2024]:
        for t in ['평일', '주말']:
            mm = m[(m.구분 == t) & (m.연도 >= start)]
            for g in G:
                r = stats.spearmanr(mm['연도'], mm[g]); yr = mm.groupby('연도')[g].mean()
                p26 = mm[mm.연도 == 2026].set_index('월')[g]; p25 = mm[mm.연도 == 2025].set_index('월')[g]
                rows.append(dict(시작=start, 구분=t, 역=g, n=len(mm), rho=round(r.statistic, 3), p=r.pvalue,
                                 판정='상승' if (r.pvalue < 0.05 and r.statistic > 0) else ('하락' if r.pvalue < 0.05 else '추세 없음'),
                                 **{str(y): round(v, 3) for y, v in yr.items()}, 이월26gt25=f'{int((p26 > p25).sum())}/6'))
    q2 = pd.DataFrame(rows); q2.to_csv(f'{OUT}/Q2_추세_Spearman.csv', index=False, encoding='utf-8-sig'); R['Q2'] = q2

    # 질문 2 가설 A: 성수권(성수+뚝섬+서울숲) 주말 승하차 중 성수역 비중
    w = nd[nd.구분 == '주말'].copy(); w['비중'] = w['성수'] / w['성수권'] * 100
    yr = w.groupby('연도')['비중'].mean(); mo = w[w.연도 >= 2023].groupby(['연도', '월'])['비중'].mean()
    r = stats.spearmanr(np.arange(len(mo)), mo.values)
    R['성수권내비중'] = dict(연도별=yr.round(1).to_dict(), rho=r.statistic, p=r.pvalue)
    pd.DataFrame({'성수권_중_성수역_비중': yr}).round(2).to_csv(f'{OUT}/Q2_성수권안_성수역비중.csv', encoding='utf-8-sig')

    # 곁가지: 금요일 ÷ 월~목 (공통 역, 노선별 역, 평일 평균 승하차 1,000명 이상)
    rows = []; sd = d[d.공통 & (d.구분 == '평일')]
    for y in [2019, 2023, 2024, 2025, 2026]:
        s = sd[sd.연도 == y]
        a = s.groupby(['노선명', '역명']).apply(lambda g: pd.Series({'평균': g.승하차.mean(), '금': g[g.요일 == 4].승하차.mean(), '월목': g[g.요일 < 4].승하차.mean()}), include_groups=False)
        a = a[a.평균 >= 1000]; ratio = a.금 / a.월목; v = ratio[('2호선', '성수')]
        xs = nd[(nd.연도 == y) & (nd.구분 == '평일')]; f = xs[xs.요일 == 4]['성수_sh']; mt = xs[xs.요일 < 4]['성수_sh']
        rows.append(dict(연도=y, 역수=len(a), 성수_금요일증가율=round(v, 3), 전체역_중앙값=round(ratio.median(), 3),
                         성수_상위퍼센트=round((ratio > v).mean() * 100 + 100 / len(a), 1), 점유율_금=round(f.mean(), 3), 점유율_월목=round(mt.mean(), 3),
                         n금=len(f), n월목=len(mt), p=stats.mannwhitneyu(f, mt).pvalue))
    fr = pd.DataFrame(rows); fr.to_csv(f'{OUT}/금요일효과.csv', index=False, encoding='utf-8-sig'); R['금요일'] = fr

    # 연결 그림: 주말÷평일 승객 비율과 역 순위 (역명 단위, 평일 평균 승차 1,000명 이상, 양평 제외)
    rows = []; e = d[(d.구분 != '제외') & (d.역명 != '양평')]
    for y in [2019, 2023, 2024, 2025, 2026]:
        mm = e[e.연도 == y].groupby(['역명', '구분', '사용일자'])[['승차총승객수', '승하차']].sum().groupby(['역명', '구분']).mean()
        a = mm['승하차'].unstack('구분'); b = mm['승차총승객수'].unstack('구분')
        a = a[b['평일'] >= 1000].dropna(); r = (a['주말'] / a['평일']).sort_values(ascending=False); k = list(r.index).index('성수') + 1
        rows.append(dict(연도=y, 역수=len(r), 성수=r['성수'], 순위=k, 상위pct=k / len(r) * 100, 중앙값=r.median(), 주말증가역수=int((r > 1).sum())))
    rk = pd.DataFrame(rows); rk.to_csv(f'{OUT}/성수_주말평일비율_순위.csv', index=False, encoding='utf-8-sig'); R['순위'] = rk
    # 2019 → 2026 주말÷평일 비율 변화 (두 해 모두 있는 역), 변화 순위
    rr = {}
    for y in [2019, 2026]:
        mm = e[e.연도 == y].groupby(['역명', '구분', '사용일자'])[['승차총승객수', '승하차']].sum().groupby(['역명', '구분']).mean()
        a = mm['승하차'].unstack('구분'); b = mm['승차총승객수'].unstack('구분'); a = a[b['평일'] >= 1000].dropna(); rr[y] = a['주말'] / a['평일']
    ch = pd.concat(rr, axis=1).dropna(); ch.columns = ['2019', '2026']; ch['변화'] = ch['2026'] - ch['2019']
    ch = ch.sort_values('변화', ascending=False); ch['변화순위'] = range(1, len(ch) + 1)
    ch.to_csv(f'{OUT}/주말평일비율_변화_2019_2026.csv', encoding='utf-8-sig'); R['변화'] = ch.head(10)
    R['daily'] = x
    return R

if __name__ == '__main__':
    R = run()
    for k, v in R.items():
        if k != 'daily': print('==', k); print(v if not isinstance(v, pd.DataFrame) else v.round(3).to_string())
