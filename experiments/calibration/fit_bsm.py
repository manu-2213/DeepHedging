"""Black-Scholes side of the calibration: one entry per year with the spot at the start of the year (S0), the
annualised mean and volatility of the daily log returns (the simulator uses the volatility; the mean is kept for the
record) and five strikes around the money (every second listed strike, at-the-money in the middle).

Writes hedging_engine/data/bsm_data_<market>_2017_2021.json (the file name is historical: sp500x holds 2017-2023).
Years already in the file are kept as they are, so the banks built from them stay valid. The 2022 and 2023 entries of
sp500x were first written rounded to four decimals.

usage (from this directory):  python fit_bsm.py <market> <year> [<year> ...]      e.g.  python fit_bsm.py sp500 2017 2018 2019 2020 2021
"""
import json
import sys
from pathlib import Path

from utils import get_BSM_data, sample_strikes

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "hedging_engine" / "data"
DATA_ALIAS = {"sp500x": "sp500"}   # market sp500x = the S&P 500 chains over 2017-2023


def main(market, years):
    out_path = OUTPUT_DIR / f"bsm_data_{market}_2017_2021.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for year in years:
        if year in results:
            print(f"  {year}: already in {out_path.name}, kept")
            continue
        S0, mean, std, K = get_BSM_data(year, DATA_ALIAS.get(market, market))
        K_sampled = sample_strikes(S0, K, 5, 2)
        results[year] = {"S0": float(S0), "mean": float(mean), "std": float(std), "K_sampled": [float(k) for k in K_sampled]}
        print(f"  {year}: S0={S0:.2f}, std={std:.4f}, sampled strikes={K_sampled}")
    out_path.write_text(json.dumps(dict(sorted(results.items())), indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
