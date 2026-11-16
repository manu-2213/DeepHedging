"""Convert OptionsDX SPX end-of-day files (one row per date/expiry/strike with call and put columns) into the yearly
long format the calibration pipeline reads: columns date, forward_price, time_to_maturity, tau, risk_free_rate, is_call,
strike_price, option_price (orient='index' JSON, bz2). Spot = UNDERLYING_LAST; price = bid/ask mid (both sides > 0);
rate = 13-week T-bill close of the day (^IRX / 100) forward-filled.

The raw files (spx_eod_YYYYMM.txt, bought from OptionsDX) are not part of the repository.
usage: python convert_spx.py <folder with the raw files> [<output folder, default option_data>]   (needs yfinance)"""
import glob, sys, numpy as np, pandas as pd, yfinance as yf
RAW = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else "option_data"
irx = yf.download("^IRX", start="2021-12-01", end="2024-01-10", progress=False, auto_adjust=True)["Close"].squeeze() / 100.0
irx.index = pd.to_datetime(irx.index).tz_localize(None)
for year in ("2022", "2023"):
    parts = []
    for f in sorted(glob.glob(f"{RAW}/spx_eod_{year}*.txt")):
        df = pd.read_csv(f, skipinitialspace=True, na_values=["", " "], low_memory=False)
        df.columns = [c.strip().strip("[]") for c in df.columns]
        parts.append(df)
    d = pd.concat(parts, ignore_index=True)
    d["date"] = pd.to_datetime(d["QUOTE_DATE"].astype(str).str.strip())
    d["expiry"] = pd.to_datetime(d["EXPIRE_DATE"].astype(str).str.strip())
    d["dte"] = (d["expiry"] - d["date"]).dt.days
    for c in ("UNDERLYING_LAST", "STRIKE", "C_BID", "C_ASK", "P_BID", "P_ASK", "C_VOLUME", "P_VOLUME"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    r = irx.reindex(pd.DatetimeIndex(sorted(d["date"].unique())), method="ffill")
    d["risk_free_rate"] = d["date"].map(r)
    rows = []
    for side, b, a, flag in (("C", "C_BID", "C_ASK", 1), ("P", "P_BID", "P_ASK", -1)):
        ok = (d[b] > 0) & (d[a] > 0) & (d["dte"] > 0)
        x = d.loc[ok, ["date", "UNDERLYING_LAST", "dte", "risk_free_rate", "STRIKE"]].copy()
        x["option_price"] = 0.5 * (d.loc[ok, b] + d.loc[ok, a]).values
        x["is_call"] = flag
        rows.append(x)
    out = pd.concat(rows, ignore_index=True).rename(columns={"UNDERLYING_LAST": "forward_price", "dte": "time_to_maturity", "STRIKE": "strike_price"})
    out["tau"] = out["time_to_maturity"] / 365.0
    out = out[["date", "forward_price", "time_to_maturity", "tau", "risk_free_rate", "is_call", "strike_price", "option_price"]].sort_values(["date", "time_to_maturity", "strike_price"]).reset_index(drop=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_json(f"{OUT}/sp500_{year}.json.bz2", orient="index", compression="bz2")
    print(year, len(out), "rows;", out["date"].nunique(), "dates; spot first", out["forward_price"].iloc[0], "; tau range", round(out["tau"].min(), 3), round(out["tau"].max(), 2),
          "; rate range", round(out["risk_free_rate"].min(), 4), round(out["risk_free_rate"].max(), 4), flush=True)
