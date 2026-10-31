# Part 1 (교수님 EDA 질문 Q1~Q5) + Part 2 (팀 아이디어 ②③, 종합)
# 데이터: 수업에서 받은 2019년 1~6월 csv 6개 (PROF_DIR). 컬럼: 사용일자, 노선명, 역명, 승차총승객수, 하차총승객수, 등록일자
# Q6~Q8(노선별)과 Part 2 ①(주말÷평일 승차 27개 역)은 노선이 구분된 공개 원본으로 계산 → s1_subway.py, s4_q2_page.py
import glob, numpy as np, pandas as pd, holidays
from scipy import stats
import scikit_posthocs as sp
from common import PROF_DIR, OUT

# 업무·상업·관광·주거 36개 역 (지표를 보기 전에 분류)
TYPES = {'업무': '강남 역삼 선릉 삼성 여의도 종각 시청 을지로입구 광화문 구로디지털단지',
         '상업': '홍대입구 신촌 이태원 명동 건대입구 합정 압구정 동대문역사문화공원',
         '관광': '경복궁 안국 대공원 어린이대공원 뚝섬유원지 여의나루 올림픽공원 경마공원',
         '주거': '상계 중계 하계 마들 수락산 방화 고덕 명일 마천 거여'}
STATION_TYPE = {s: k for k, v in TYPES.items() for s in v.split()}
SEOL = [pd.Timestamp(2019, 2, d) for d in (4, 5, 6)]

def load():
    d = pd.concat([pd.read_csv(f, encoding='utf-8-sig') for f in sorted(glob.glob(f'{PROF_DIR}/*.csv'))])
    d['날짜'] = pd.to_datetime(d.사용일자.astype(str))
    d['승하차'] = d.승차총승객수 + d.하차총승객수
    return d

def run():
    d = load()
    # 요일은 사용일자로 다시 계산 (pandas: 0=월요일). 예시 코드의 {0:'일', ...} 매핑은 하루씩 밀려 있었음
    hol = {pd.Timestamp(x) for x in holidays.KR(years=2019)}
    day = d.groupby('날짜').승하차.sum().to_frame()
    day['월'] = day.index.month; day['요일'] = day.index.dayofweek; day['공휴일'] = day.index.isin(hol)
    res = {}

    # Q1·Q2 가장 많은 달 (181일 전체)
    m = day.groupby('월').승하차
    res['월총합'] = m.sum(); res['하루중앙값'] = m.median(); res['하루평균'] = m.mean()
    res['Q2_KW'] = stats.kruskal(*[g for _, g in m])
    res['Q2_Dunn'] = sp.posthoc_dunn(day, val_col='승하차', group_col='월', p_adjust='bonferroni')
    res['Q2_5월vs나머지'] = stats.mannwhitneyu(day[day.월 == 5].승하차, day[day.월 != 5].승하차)

    # Q3 요일 (평일 공휴일 제외)
    w = day[~(day.공휴일 & (day.요일 < 5))]
    res['Q3_KW'] = stats.kruskal(*[g for _, g in w.groupby('요일').승하차])
    res['Q3_Dunn'] = sp.posthoc_dunn(w, val_col='승하차', group_col='요일', p_adjust='bonferroni')
    wk = w[w.요일 < 5]; res['Q3_월to금_rho'] = stats.spearmanr(wk.요일, wk.승하차)

    # Q4 평일 공휴일 vs 근무일
    ph = day[day.공휴일 & (day.요일 < 5)].승하차; work = day[~day.공휴일 & (day.요일 < 5)].승하차
    res['Q4'] = (len(ph), len(work), ph.median(), work.median(), stats.mannwhitneyu(ph, work))

    # Q5 가장 많이 타는 역 (6개월 승차 합계), 1·2위는 같은 날짜끼리 Wilcoxon
    on = d.groupby(['날짜', '역명']).승차총승객수.sum().unstack()
    res['Q5_합계'] = on.sum().sort_values(ascending=False)
    res['Q5_잠실vs강남'] = stats.wilcoxon(on['잠실'], on['강남'])
    we = on[on.index.dayofweek >= 5]
    res['Q5_평일평균'] = on[(on.index.dayofweek < 5) & ~on.index.isin(hol)].mean().sort_values(ascending=False)
    res['Q5_주말평균'] = we.mean().sort_values(ascending=False)
    diff = we['고속터미널'] - we['홍대입구']
    res['Q5_주말12'] = (stats.wilcoxon(we['고속터미널'], we['홍대입구']), stats.shapiro(diff), stats.ttest_rel(we['고속터미널'], we['홍대입구']))

    # Part 2 ② 역 유형별 주말/평일 비율 (승하차, 주말 하루 평균 ÷ 공휴일 뺀 평일 하루 평균)
    st = d.groupby(['날짜', '역명']).승하차.sum().unstack()
    dow = st.index.dayofweek; ish = st.index.isin(hol)
    wk_mean = st[(dow < 5) & ~ish].mean()
    ratio = st[dow >= 5].mean() / wk_mean
    r36 = ratio[list(STATION_TYPE)]
    grp = pd.Series(STATION_TYPE)
    res['P2_유형KW'] = stats.kruskal(*[r36[grp[grp == k].index] for k in TYPES])
    res['P2_유형중앙값'] = r36.groupby(grp).median()
    res['P2_유형Dunn'] = sp.posthoc_dunn(pd.DataFrame({'r': r36, 'g': grp}), val_col='r', group_col='g', p_adjust='bonferroni')

    # Part 2 ③ 설 연휴 vs 다른 평일 공휴일 감소율 = 1 - 그날 승객 ÷ 그 역 평소 평일 하루 평균 (양평 제외)
    other = sorted(x for x in hol if x.dayofweek < 5 and x.month <= 6 and x not in SEOL)
    ok = wk_mean[(wk_mean > 0) & (wk_mean.index != '양평')].index
    seol = (1 - st.loc[SEOL, ok].div(wk_mean[ok])).mean()
    oth = (1 - st.loc[other, ok].div(wk_mean[ok])).mean()
    v = seol.notna() & oth.notna()
    seol, oth = seol[v], oth[v]
    res['P2_설'] = (len(seol), seol.median(), oth.median(), (seol > oth).mean(), stats.wilcoxon(seol, oth))

    # 종합: 36개 역 주말 비율 순위 vs 설 감소율 순위 (주말에 많이 빌수록 설에도 많이 비나)
    s36 = list(STATION_TYPE)
    res['P2_종합_spearman'] = stats.spearmanr(-ratio[s36], seol[s36])
    res['P2_종합_pearson'] = stats.pearsonr(-ratio[s36], seol[s36])

    pd.DataFrame({'월총합': res['월총합'], '하루중앙값': res['하루중앙값'], '하루평균': res['하루평균']}).to_csv(f'{OUT}/P1_월별.csv', encoding='utf-8-sig')
    res['Q5_합계'].rename('승차합계').to_csv(f'{OUT}/P1_역별_승차합계.csv', encoding='utf-8-sig')
    pd.DataFrame({'유형': grp, '주말평일비': r36}).to_csv(f'{OUT}/P2_36개역_유형별_비율.csv', encoding='utf-8-sig')
    pd.DataFrame({'설감소율': seol, '다른공휴일감소율': oth}).to_csv(f'{OUT}/P2_설연휴_감소율.csv', encoding='utf-8-sig')
    return res

def verify(res):
    ok, bad = 0, []
    def chk(name, got, want, nd):
        nonlocal ok
        if abs(round(float(got), nd) - want) < 1e-9: ok += 1
        else: bad.append(f'{name}: 계산 {round(float(got), nd)} / 발표 {want}')
    chk('5월 총합(억)', res['월총합'][5] / 1e8, 4.86, 2); chk('4월 총합(억)', res['월총합'][4] / 1e8, 4.71, 2)
    chk('5월 중앙값(만)', res['하루중앙값'][5] / 1e4, 1710, 0)
    chk('4월 평균(만)', res['하루평균'][4] / 1e4, 1570, 0); chk('5월 평균(만)', res['하루평균'][5] / 1e4, 1567, 0)
    chk('5월 vs 나머지 p', res['Q2_5월vs나머지'].pvalue, 0.003, 3)
    for k, v in [(3, 1.00), (4, 1.00), (6, 0.41)]: chk(f'Dunn 5 vs {k}', res['Q2_Dunn'].loc[5, k], v, 2)
    chk('금 vs 목 p', res['Q3_Dunn'].loc[4, 3], 1.00, 2); chk('월~금 ρ', res['Q3_월to금_rho'][0], 0.46, 2)
    n_ph, n_w, m_ph, m_w, u = res['Q4']
    chk('평일 공휴일 수', n_ph, 7, 0); chk('근무일 수', n_w, 122, 0); chk('공휴일 중앙값(만)', m_ph / 1e4, 746, 0); chk('근무일 중앙값(만)', m_w / 1e4, 1676, 0); chk('U', u.statistic, 0, 0)
    chk('잠실 합계(만)', res['Q5_합계']['잠실'] / 1e4, 1857, 0); chk('잠실vs강남 p', res['Q5_잠실vs강남'].pvalue, 0.27, 2)
    if res['Q5_합계'].index[0] == '잠실' and res['Q5_평일평균'].index[0] == '강남' and list(res['Q5_주말평균'].index[:2]) == ['고속터미널', '홍대입구']: ok += 1
    else: bad.append('Q5 1위 순서')
    wx, sh, tt = res['Q5_주말12']; chk('주말 1·2위 Wilcoxon p', wx.pvalue, 0.04, 2); chk('차이 정규성 p', sh.pvalue, 0.19, 2); chk('주말 1·2위 t p', tt.pvalue, 0.065, 3)
    chk('유형 H', res['P2_유형KW'].statistic, 26.06, 2)
    n, s, o, frac, _ = res['P2_설']
    chk('설 역 수', n, 508, 0); chk('설 감소율', s * 100, 63, 0); chk('다른 공휴일 감소율', o * 100, 42, 0); chk('설이 더 큰 역 비율', frac * 100, 99, 0)
    chk('종합 ρ', res['P2_종합_spearman'][0], 0.56, 2); chk('Pearson r', abs(res['P2_종합_pearson'][0]), 0.06, 2)
    print(f'Part 1·2: 확인 {ok}개 일치, 불일치 {len(bad)}개'); [print(' -', b) for b in bad]
    return bad

if __name__ == '__main__':
    verify(run())
