# 공통 설정: 경로, 공휴일, 지역 코드
import os, pandas as pd, holidays
HERE = os.path.dirname(os.path.abspath(__file__))

SUBWAY_DIR = os.environ.get('SUBWAY_DIR', os.path.join(HERE, 'data', 'subway'))            # CARD_SUBWAY_MONTH_*.csv
LM_DIR     = os.environ.get('LM_DIR', os.path.join(HERE, 'data', 'living'))           # seoul_purpose_admdong1_in_YYYYMM.zip (도착지, 목적)
OD_DIR     = os.environ.get('OD_DIR', os.path.join(HERE, 'data', 'od'))           # seoul_purpose_admdong4_in_YYYYMMDD.zip (출발-도착지)
OUT        = os.environ.get('OUT', os.path.join(HERE, 'out'))
CACHE      = os.environ.get('CACHE', os.path.join(HERE, 'cache'))
os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)

# 성수동 = 성수1가1동·1가2동·2가1동·2가3동, 비교 대상 화양동(건대입구)
PROF_DIR   = os.environ.get('PROF_DIR', os.path.join(HERE, 'data', 'professor'))  # 수업에서 받은 2019년 1~6월 csv 6개

SEONGSU = {'11200650': '성수1가1동', '11200660': '성수1가2동', '11200670': '성수2가1동', '11200690': '성수2가3동'}
HWAYANG = {'11215710': '화양동'}

def holiday_set(years):
    """공휴일(대체·선거·임시 포함) + 매년 5월 1일"""
    s = set()
    for y in years:
        s |= {pd.Timestamp(x) for x in holidays.KR(years=y)}
        s.add(pd.Timestamp(y, 5, 1))
    return s

def day_type(dt, hol):
    if dt in hol: return '제외'
    return '주말' if dt.dayofweek >= 5 else '평일'

def holm(ps):
    import numpy as np
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * ps[j]); adj[j] = min(1, run)
    return adj
