# 2026-IoT · FedACG 공격·방어 연구 (mobicom)

서버 모멘텀을 쓰는 FedACG가 공격에 더 취약한지, 강건 집계(Krum, Multi-Krum, Trimmed Mean, Median)를 붙였을 때 효율과 보안을 함께 지킬 수 있는지 검증한다.

- 구현 기반: [FedACG (CVPR 2024)](https://github.com/geehokim/FedACG)
- 서베이: The Impact of Adversarial Attacks on Federated Learning: A Survey (IEEE TPAMI 2024)

## 작업 규칙
- 각자 작업 브랜치에서 작업하고 PR로 합친다.
- 실험 설정은 설정 파일에 모으고 코드 안에 숫자를 직접 쓰지 않는다.
- 결과 파일 이름: `알고리즘_집계_공격_비율_시드.csv` (예: `fedacg_krum_lf_20_s1.csv`)
