# 전체 재현: Part 1·2(수업 파일) → 1단계(지하철) → 2단계(생활이동 도착지) → 3단계(출발-도착지) → 그림 → 숫자 검증
# 데이터 위치는 환경변수로: PROF_DIR, SUBWAY_DIR, LM_DIR, OD_DIR (common.py 참고)
import p0_part1_part2, p1_q6_q8, s1_subway, s2_living, s3_od, s4_q2_page, figures, verify
if __name__ == '__main__':
    p0_part1_part2.verify(p0_part1_part2.run()); p1_q6_q8.verify(p1_q6_q8.run())
    s1_subway.run(); s2_living.run(); s3_od.run(); s4_q2_page.run()
    for k in range(1, 13): getattr(figures, f'fig{k}')()
    verify.main()
