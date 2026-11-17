"""결과 파일 이름 규칙: 알고리즘_집계_공격_비율_시드.csv (예: fedacg_krum_lf_20_s1.csv)."""


def result_filename(algorithm: str, aggregator: str, attack: str, ratio_pct: int, seed: int) -> str:
    parts = [algorithm.lower(), aggregator.lower().replace("-", "").replace("_", ""),
             attack.lower(), str(int(ratio_pct)), f"s{int(seed)}"]
    return "_".join(parts) + ".csv"
