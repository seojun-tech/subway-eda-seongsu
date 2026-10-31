# Part 1 Q6~Q8 (노선별 질문): 노선이 구분된 공개 원본 CARD_SUBWAY_MONTH_2019.csv (SUBWAY_DIR)로 계산
# 수업 파일은 역마다 노선명을 하나로 통일해 두어 환승역의 노선을 나눌 수 없었음.
# 공개 원본 2019년 1~6월은 수업 파일과 행 수(107,028개)·순서·승하차 합계가 같고, 노선명만 구분돼 있음 (check_same)
import glob, pandas as pd, holidays
from common import SUBWAY_DIR, PROF_DIR, OUT

# 노선 묶음: 1호선 = 경부·경원·경인·장항선, 3호선 = 일산선, 4호선 = 과천·안산선, 9호선 = 2~3단계 포함
LINE = {'경부선': '1호선', '경원선': '1호선', '경인선': '1호선', '장항선': '1호선', '일산선': '3호선',
        '과천선': '4호선', '안산선': '4호선', '9호선2~3단계': '9호선'}
LINES = [f'{i}호선' for i in range(1, 10)]
DAYS = ['월', '화', '수', '목', '금', '토', '일']

def load():
    d = pd.read_csv(f'{SUBWAY_DIR}/CARD_SUBWAY_MONTH_2019.csv', encoding='cp949', index_col=False, usecols=range(6))
    d['날짜'] = pd.to_datetime(d.사용일자.astype(str))
    d = d[d.날짜.dt.month <= 6].reset_index(drop=True)
    d['노선'] = d.노선명.replace(LINE)
    d['역'] = d.역명.str.replace(r'\(.*\)', '', regex=True)          # 잠실(송파구청) → 잠실
    d['요일'] = d.날짜.dt.dayofweek                                    # 0=월
    return d

def check_same(d):
    """수업 파일과 공개 원본이 같은 데이터인지: 행 수, 날짜·역 순서, 승하차 인원"""
    fs = sorted(glob.glob(f'{PROF_DIR}/*.csv'))
    if not fs: return None
    p = pd.concat([pd.read_csv(f, encoding='utf-8-sig') for f in fs]).sort_values('사용일자', kind='stable').reset_index(drop=True)  # 월 파일 순서대로
    return (len(p) == len(d) and (p.사용일자.astype(str).values == d.사용일자.astype(str).values).all()
            and (p.역명.values == d.역.values).all()
            and (p.승차총승객수.values == d.승차총승객수.values).all() and (p.하차총승객수.values == d.하차총승객수.values).all())

def run():
    d = load(); x = d[d.노선.isin(LINES)]
    hol = {pd.Timestamp(t) for t in holidays.KR(years=2019)}
    res = {'같은데이터': check_same(d), '행수': len(d)}
    # Q6 노선 × 요일 승차 (6개월 합계)
    hm = x.pivot_table(index='노선', columns='요일', values='승차총승객수', aggfunc='sum'); hm.columns = DAYS
    res['Q6'] = hm
    # Q6 노선별 승차 상위 3개 역 × 요일: 공휴일 제외 하루 평균, 그 역 평일 평균 대비 배수
    y = x[~x.날짜.isin(hol)]; rows = []
    for ln in LINES:
        s = y[y.노선 == ln]
        for st in s.groupby('역').승차총승객수.sum().nlargest(3).index:
            m = s[s.역 == st].groupby('날짜').승차총승객수.sum(); m = m.groupby(m.index.dayofweek).mean()
            rows.append({'노선': ln, '역': st, **{DAYS[i]: m[i] for i in range(7)}, **{f'{DAYS[i]}_배수': m[i] / m[:5].mean() for i in range(7)}})
    res['Q6_역'] = pd.DataFrame(rows)
    # Q7 1호선 하차, Q8 2호선 승차 (환승역은 그 노선 몫만)
    res['Q7'] = x[x.노선 == '1호선'].groupby('역').하차총승객수.sum().sort_values(ascending=False)
    res['Q8'] = x[x.노선 == '2호선'].groupby('역').승차총승객수.sum().sort_values(ascending=False)
    hm.to_csv(f'{OUT}/P1_Q6_노선x요일_승차.csv', encoding='utf-8-sig')
    res['Q6_역'].to_csv(f'{OUT}/P1_Q6_노선별_상위3역_요일.csv', encoding='utf-8-sig', index=False)
    res['Q7'].rename('하차합계').to_csv(f'{OUT}/P1_Q7_1호선_하차.csv', encoding='utf-8-sig')
    res['Q8'].rename('승차합계').to_csv(f'{OUT}/P1_Q8_2호선_승차.csv', encoding='utf-8-sig')
    return res

def verify(res):
    ok, bad = 0, []
    def chk(name, got, want, nd=0):
        nonlocal ok
        if abs(round(float(got), nd) - want) < 1e-9: ok += 1
        else: bad.append(f'{name}: 계산 {round(float(got), nd)} / 발표 {want}')
    if res['같은데이터'] is not None:
        if res['같은데이터']: ok += 1
        else: bad.append('수업 파일과 공개 원본이 다름')
    chk('행 수', res['행수'], 107028)
    hm = res['Q6'] / 1e4                                               # 10장 히트맵 (만 명)
    want = {'1호선': [3450, 3608, 3676, 3736, 3897, 3279, 2473], '2호선': [4022, 4250, 4360, 4468, 4652, 3508, 2472],
            '3호선': [1783, 1875, 1926, 1965, 2067, 1599, 1107], '4호선': [2126, 2242, 2306, 2337, 2435, 1960, 1445],
            '5호선': [1641, 1729, 1770, 1791, 1839, 1296, 903], '6호선': [933, 998, 1012, 1028, 1057, 812, 587],
            '7호선': [1950, 2043, 2077, 2106, 2152, 1544, 1114], '8호선': [515, 542, 553, 562, 578, 416, 303],
            '9호선': [940, 992, 1013, 1036, 1083, 769, 551]}
    for ln, v in want.items():
        for day, w in zip(DAYS, v): chk(f'Q6 {ln} {day}', hm.loc[ln, day], w)
    chk('일요일 1호선 − 2호선', res['Q6'].loc['1호선', '일'] - res['Q6'].loc['2호선', '일'], 4657)
    r = res['Q6_역'].set_index('역')                                   # 11장
    for st, w in [('가산디지털단지', 0.38), ('문정', 0.44), ('여의도', 0.41)]: chk(f'{st} 토요일 배수', r.loc[st, '토_배수'], w, 2)
    for st in ['홍대입구', '고속터미널', '이태원']:
        if r.loc[st, DAYS].astype(float).idxmax() == '토': ok += 1
        else: bad.append(f'{st} 토요일 최다 아님')
    for i, (st, w) in enumerate([('서울역', 1000), ('영등포', 943), ('수원', 908)]):      # 12장
        chk(f'Q7 {i + 1}위 {st}', res['Q7'][st] / 1e4, w)
        if res['Q7'].index[i] == st: ok += 1
        else: bad.append(f'Q7 {i + 1}위 {res["Q7"].index[i]}')
    for i, (st, w) in enumerate([('강남', 1815), ('잠실', 1569), ('홍대입구', 1468)]):     # 12·13장
        chk(f'Q8 {i + 1}위 {st}', res['Q8'][st] / 1e4, w)
        if res['Q8'].index[i] == st: ok += 1
        else: bad.append(f'Q8 {i + 1}위 {res["Q8"].index[i]}')
    chk('2호선 역 수', len(res['Q8']), 50)
    print(f'Q6~Q8: 확인 {ok}개 일치, 불일치 {len(bad)}개'); [print(' -', b) for b in bad]
    return bad

if __name__ == '__main__':
    verify(run())
