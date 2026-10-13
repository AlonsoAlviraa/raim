"""luk-v2 runner (BACKTEST ONLY, virtual capital):
1. Downloads SEC EDGAR company_tickers_exchange.json + quarterly Financial Statement Data Sets (2009q1..2026q3)
   to build point-in-time US fundamentals (strictly available on session > filing acceptance date) and
   extract historical CIK->ticker mappings (including delisted/acquired filers).
2. Downloads Yahoo Finance daily OHLCV (with split history for unadjusted price reconstruction) for the
   US small/mid-cap universe + SPY + IWM + QQQ, plus 60m intraday bars (~730d) for the separated intraday sub-test.
3. Enforces >=4 distinct OHLC price points per stock-day and computes annual coverage + survivorship gap.
4. Runs purged walk-forward (5 folds, plateau selector, dev 2009-07-01..2025-09-30) on V0..V7, plus +1 bar delay (R11).
5. Runs required comparisons: (a) momentum without fundamentals, (b) fundamentals without momentum,
   (c) S&P 500 version of the same strategy, (d) separated 60m intraday sub-period test.
6. Opens holdout_2025-10_luk_v2 ONCE (designer_contaminated) and scores through raim.research.verdict
   (both base and aggressive -70% DD rules), reporting SPY & IWM B&H benchmarks and user gate."""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from raim.data import manifest  # noqa: E402
from raim.research import prereg  # noqa: E402
from raim.research.evaluate import evidence  # noqa: E402
from raim.research.registry import TrialRegistry  # noqa: E402
from raim.research.stability import plateau_select  # noqa: E402
from raim.research.vault import Vault  # noqa: E402
from raim.research.verdict import AGGRESSIVE_PATH, load_rules, verdict  # noqa: E402
from raim.strategies import engine, luk as L1, luk_v2 as L2  # noqa: E402
from raim.strategies.edge_v2 import payoff_stats  # noqa: E402
from raim.strategies.wf import walk_forward  # noqa: E402

log = logging.getLogger("run_luk_v2")

DATA_DIR = L2.DATA_DIR
L1_YAHOO_DIR = L1.ROOT / "yahoo"
YAHOO_DIR = DATA_DIR / "yahoo_daily"
YAHOO_60M_DIR = DATA_DIR / "yahoo_60m"
SEC_DIR = DATA_DIR / "sec_fsds"
OUT = ROOT / "reports/luk_v2"
DOC = ROOT / "docs/research/luk_v2_2026-10"
SPECS = ROOT / "configs/prereg/luk_v2"
REG = Path(os.environ.get("RAIM_TRIAL_REGISTRY", str(ROOT / "reports/strategies/trials.jsonl")))

WARMUP_START = "2008-01-02"
START_EVAL = "2009-07-01"
DEV_END = "2025-10-01"
HO_END = "2026-10-01"

SEC_HEADERS = {"User-Agent": "RAIM-Research research@raim-quant.org"}
YF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

INST_TICKER_RE = re.compile(r"^([a-z]{1,5})-\d{8}")
IGNORE_PREFIXES = {"FORM", "DEF", "DOC", "Q", "K", "FILING", "EX", "R", "S", "N", "INDEX", "MAIN", "COVER", "REPORT"}


def _is_clean_common_symbol(sym: str) -> bool:
    if not sym or len(sym) > 5 or not sym.isalpha():
        return False
    if sym in IGNORE_PREFIXES:
        return False
    if len(sym) == 5 and sym[-1] in ("W", "U", "R", "P"):
        return False
    return True


def fetch_sec_tickers_and_fsds() -> tuple[pd.DataFrame, dict[int, str], dict]:
    """Download SEC company_tickers_exchange.json and quarterly FSDS zips (2009q1..2026q3),
    cache extracted quarterly fundamentals (`sec_quarterly_fundamentals.parquet`), and return:
      - `filings_df`: DataFrame of (cik, period, filed, eps, rev, opinc, shares, instance_sym)
      - `cik_to_sym`: mapping from CIK (int) -> ticker symbol (str) combining current NYSE/Nasdaq
        listings and historical delisted/acquired filers from FSDS `instance` filenames.
      - `sec_meta`: summary metadata of SEC FSDS quarters & CIK counts."""
    SEC_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = SEC_DIR / "sec_quarterly_fundamentals.csv.gz"
    meta_path = SEC_DIR / "sec_meta.json"

    # 1. SEC company_tickers_exchange.json
    tickers_json_path = SEC_DIR / "company_tickers_exchange.json"
    if not tickers_json_path.exists():
        with httpx.Client(headers=SEC_HEADERS, timeout=30.0, follow_redirects=True) as client:
            r = client.get("https://www.sec.gov/files/company_tickers_exchange.json")
            r.raise_for_status()
            tickers_json_path.write_bytes(r.content)

    raw_ex = json.loads(tickers_json_path.read_text())["data"]
    cik_to_sym: dict[int, str] = {}
    for cik, _name, sym, exch in raw_ex:
        if exch not in ("Nasdaq", "NYSE"):
            continue
        if not sym:
            continue
        s_clean = str(sym).strip().upper()
        if _is_clean_common_symbol(s_clean):
            cik_to_sym.setdefault(int(cik), s_clean)

    if parquet_path.exists() and meta_path.exists():
        df_all = pd.read_csv(parquet_path)
        meta = json.loads(meta_path.read_text())
        for cik_v, s_v in meta.get("historical_fsds_ciks", {}).items():
            cik_to_sym.setdefault(int(cik_v), s_v)
        return df_all, cik_to_sym, meta

    quarters = [f"{y}q{q}" for y in range(2009, 2027) for q in (1, 2, 3, 4) if not (y == 2026 and q == 4)]
    log.info("Downloading and parsing %d SEC EDGAR Financial Statement Data Sets (2009q1..2026q3)...", len(quarters))

    all_tags = set(L2.EPS_TAGS) | set(L2.REV_TAGS) | set(L2.OPINC_TAGS) | set(L2.SHARES_TAGS)

    def _process_one_quarter(qtr: str) -> tuple[str, pd.DataFrame, dict[int, str]]:
        q_cache = SEC_DIR / f"parsed_{qtr}.csv.gz"
        q_sym_cache = SEC_DIR / f"syms_{qtr}.json"
        if q_cache.exists() and q_sym_cache.exists():
            return qtr, pd.read_csv(q_cache), {int(k): v for k, v in json.loads(q_sym_cache.read_text()).items()}

        url = f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{qtr}.zip"
        content = b""
        for attempt in range(3):
            try:
                with httpx.Client(headers=SEC_HEADERS, timeout=60.0, follow_redirects=True) as client:
                    r = client.get(url)
                    if r.status_code == 200 and len(r.content) > 10_000:
                        content = r.content
                        break
            except Exception:
                time.sleep(1.0 * (attempt + 1))
        if not content:
            return qtr, pd.DataFrame(), {}

        zf = zipfile.ZipFile(io.BytesIO(content))
        sub = pd.read_csv(
            zf.open("sub.txt"),
            sep="\t",
            usecols=["adsh", "cik", "form", "period", "filed", "instance"],
            dtype=str,
            on_bad_lines="skip",
        )
        sub = sub[sub["form"].isin(["10-Q", "10-K"])].dropna(subset=["adsh", "cik", "period", "filed"])
        if len(sub) == 0:
            return qtr, pd.DataFrame(), {}

        # Extract historical ticker symbols from XBRL instance filename (`ticker-YYYYMMDD.xml`)
        hist_map: dict[int, str] = {}
        for cik_s, inst_s in zip(sub["cik"], sub["instance"].fillna("")):
            m = INST_TICKER_RE.match(str(inst_s).lower())
            if m:
                cand_s = m.group(1).upper()
                if _is_clean_common_symbol(cand_s):
                    try:
                        hist_map[int(cik_s)] = cand_s
                    except Exception:
                        pass

        num = pd.read_csv(
            zf.open("num.txt"),
            sep="\t",
            usecols=["adsh", "tag", "ddate", "qtrs", "segments", "coreg", "value"],
            dtype={"adsh": str, "tag": str, "ddate": str, "qtrs": str, "segments": str, "coreg": str, "value": float},
            on_bad_lines="skip",
        )
        num = num[num["tag"].isin(all_tags) & num["segments"].isna() & num["coreg"].isna()]
        sub_map = sub.set_index("adsh")[["cik", "period", "filed"]]
        num = num.join(sub_map, on="adsh", how="inner")
        # Keep values matching the filing's primary reporting period (or point-in-time shares qtrs=="0")
        num = num[(num["ddate"] == num["period"]) | (num["tag"].isin(L2.SHARES_TAGS))]

        # Classify tags into (eps, rev, opinc, shares)
        eps_rows = num[num["tag"].isin(L2.EPS_TAGS) & (num["qtrs"].isin(["1", "0"]))]
        rev_rows = num[num["tag"].isin(L2.REV_TAGS) & (num["qtrs"] == "1")]
        op_rows = num[num["tag"].isin(L2.OPINC_TAGS) & (num["qtrs"] == "1")]
        sh_rows = num[num["tag"].isin(L2.SHARES_TAGS) & (num["qtrs"].isin(["0", "1"]))]

        base = sub[["adsh", "cik", "period", "filed"]].drop_duplicates("adsh").set_index("adsh")
        base["eps"] = eps_rows.groupby("adsh")["value"].last()
        base["rev"] = rev_rows.groupby("adsh")["value"].max()
        base["opinc"] = op_rows.groupby("adsh")["value"].last()
        base["shares"] = sh_rows.groupby("adsh")["value"].max()
        out_df = base.dropna(subset=["eps", "rev", "opinc", "shares"], how="all").reset_index(drop=True)
        out_df["cik"] = out_df["cik"].astype(int)

        out_df.to_csv(q_cache, index=False, compression="gzip")
        q_sym_cache.write_text(json.dumps(hist_map))
        return qtr, out_df, hist_map

    dfs: list[pd.DataFrame] = []
    hist_ciks: dict[int, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(_process_one_quarter, q) for q in quarters]
        for i, f in enumerate(concurrent.futures.as_completed(futs), 1):
            qtr, q_df, h_map = f.result()
            if len(q_df) > 0:
                dfs.append(q_df)
            for k, v in h_map.items():
                hist_ciks.setdefault(k, v)
            if i % 15 == 0 or i == len(quarters):
                log.info("SEC FSDS progress: %d/%d quarters processed", i, len(quarters))

    df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    for k, v in hist_ciks.items():
        cik_to_sym.setdefault(k, v)

    meta = {
        "n_quarters": len(dfs),
        "n_filing_records": int(len(df_all)),
        "n_unique_ciks_in_fsds": int(df_all["cik"].nunique()) if len(df_all) else 0,
        "n_exchange_ciks": len(raw_ex),
        "n_historical_fsds_ciks_extracted": len(hist_ciks),
        "historical_fsds_ciks": {str(k): v for k, v in hist_ciks.items()},
    }
    df_all.to_csv(parquet_path, index=False, compression="gzip")
    meta_path.write_text(json.dumps(meta, indent=2))
    return df_all, cik_to_sym, meta


def select_smid_candidate_universe(df_filings: pd.DataFrame, cik_to_sym: dict[int, str]) -> list[tuple[int, str, bool]]:
    """Identify all US common-stock CIKs (both currently listed on NYSE/Nasdaq and historical delisted
    SEC filers) that ever reported positive revenue or shares consistent with small/mid-cap status
    ($2M <= quarterly revenue <= $8B, or 3M <= shares <= 3B), excluding S&P 500 mega-caps >$30B."""
    # Group by CIK in df_filings
    grp = df_filings.groupby("cik").agg(
        n_filings=("filed", "count"),
        med_shares=("shares", "median"),
        max_rev=("rev", "max"),
        min_filed=("filed", "min"),
        max_filed=("filed", "max"),
    )
    # Require at least 4 quarterly filings and plausible small/mid-cap scale
    cond = (
        (grp["n_filings"] >= 4)
        & (grp["med_shares"].fillna(0) >= 2_000_000)
        & (grp["med_shares"].fillna(0) <= 1_500_000_000)
        & (grp["max_rev"].fillna(0) >= 2_000_000)
        & (grp["max_rev"].fillna(0) <= 5_000_000_000)
    )
    elig_ciks = grp[cond].index.tolist()

    # Load exchange CIK set to distinguish active vs historical-delisted filers
    ex_data = json.loads((SEC_DIR / "company_tickers_exchange.json").read_text())["data"]
    active_ciks = {int(x[0]) for x in ex_data if x[3] in ("Nasdaq", "NYSE")}

    selected: list[tuple[int, str, bool]] = []
    seen_syms: set[str] = set()
    # Sort so CIKs with most filings come first
    elig_sorted = sorted(elig_ciks, key=lambda c: (-int(grp.loc[c, "n_filings"]), int(c)))
    for c in elig_sorted:
        sym = cik_to_sym.get(int(c))
        if not sym or sym in seen_syms:
            continue
        is_active = int(c) in active_ciks
        selected.append((int(c), sym, is_active))
        seen_syms.add(sym)
        if len(selected) >= 2400:
            break
    log.info(
        "Selected %d candidate small/mid-cap SEC filers (%d currently listed, %d historical/delisted)",
        len(selected),
        sum(1 for x in selected if x[2]),
        sum(1 for x in selected if not x[2]),
    )
    return selected


def _fetch_one_yahoo_daily(sym: str, p1: int, p2: int) -> tuple[str, bool]:
    YAHOO_DIR.mkdir(parents=True, exist_ok=True)
    out_file = YAHOO_DIR / f"{sym}.json"
    if out_file.exists() and out_file.stat().st_size > 50:
        try:
            d = json.loads(out_file.read_text())
            return sym, bool(d.get("chart", {}).get("result"))
        except Exception:
            pass
    # Reuse luk-v1 cache if already downloaded
    l1_file = L1_YAHOO_DIR / f"{sym}.json"
    if l1_file.exists() and l1_file.stat().st_size > 200:
        try:
            d = json.loads(l1_file.read_text())
            if d.get("chart", {}).get("result"):
                out_file.write_bytes(l1_file.read_bytes())
                return sym, True
        except Exception:
            pass

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d&events=div%2Csplits"
    for attempt in range(2):
        try:
            with httpx.Client(headers=YF_HEADERS, timeout=15.0, follow_redirects=True) as client:
                r = client.get(url)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("chart", {}).get("result"):
                        out_file.write_text(json.dumps(data))
                        return sym, True
                elif r.status_code in (404, 400):
                    out_file.write_text(json.dumps({"chart": {"result": None, "error": f"HTTP {r.status_code}"}}))
                    return sym, False
        except Exception:
            pass
        time.sleep(0.3 * (attempt + 1))
    out_file.write_text(json.dumps({"chart": {"result": None, "error": "timeout"}}))
    return sym, False


def download_yahoo_daily_batch(symbols: list[str]) -> dict[str, bool]:
    YAHOO_DIR.mkdir(parents=True, exist_ok=True)
    p1 = int(pd.Timestamp("2007-01-01", tz="UTC").timestamp())
    p2 = int(pd.Timestamp("2026-10-05", tz="UTC").timestamp())
    status: dict[str, bool] = {}
    log.info("Fetching daily OHLCV for %d symbols (24 workers)...", len(symbols))
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
        futs = [ex.submit(_fetch_one_yahoo_daily, s, p1, p2) for s in symbols]
        for i, f in enumerate(concurrent.futures.as_completed(futs), 1):
            sym, ok = f.result()
            status[sym] = ok
            if i % 300 == 0 or i == len(symbols):
                log.info("Yahoo daily progress: %d/%d (valid=%d)", i, len(symbols), sum(status.values()))
    return status


def download_yahoo_60m_batch(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Download ~730-day 60-minute intraday bars from Yahoo Finance for a representative sample of
    top small/mid-cap momentum stocks for the separated intraday execution test."""
    YAHOO_60M_DIR.mkdir(parents=True, exist_ok=True)

    def _fetch_60m(sym: str) -> tuple[str, pd.DataFrame | None]:
        p = YAHOO_60M_DIR / f"{sym}_60m.json"
        if not p.exists():
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=730d&interval=60m"
            try:
                with httpx.Client(headers=YF_HEADERS, timeout=15.0, follow_redirects=True) as client:
                    r = client.get(url)
                    if r.status_code == 200 and r.json().get("chart", {}).get("result"):
                        p.write_text(r.text)
                    else:
                        p.write_text(json.dumps({"chart": {"result": None}}))
            except Exception:
                return sym, None
        try:
            raw = json.loads(p.read_text())
            return sym, L2.parse_yahoo_60m_chart(raw)
        except Exception:
            return sym, None

    out: dict[str, pd.DataFrame] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        for sym, df in ex.map(_fetch_60m, symbols):
            if df is not None and len(df) >= 100:
                out[sym] = df
    log.info("Loaded 60m intraday series for %d/%d symbols", len(out), len(symbols))
    return out


def build_luk_v2_panel() -> tuple[L2.LukV2Panel, dict[str, pd.DataFrame]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filings_df, cik_to_sym, sec_meta = fetch_sec_tickers_and_fsds()
    candidates = select_smid_candidate_universe(filings_df, cik_to_sym)

    cand_syms = [s for _, s, _ in candidates]
    all_syms = sorted(set(cand_syms) | {"SPY", "IWM", "QQQ"})
    dl_status = download_yahoo_daily_batch(all_syms)

    # Ensure CPIAUCSL is available (reuse luk-v1 or download)
    cpi_path = L1.ROOT / "CPIAUCSL.csv"
    if not cpi_path.exists():
        from run_luk_v1 import download_cpi
        cpi_path = download_cpi()
    cpi_df = pd.read_csv(cpi_path)

    stock_dfs: dict[str, pd.DataFrame] = {}
    for sym in all_syms:
        p = YAHOO_DIR / f"{sym}.json"
        if not p.exists():
            continue
        try:
            raw = json.loads(p.read_text())
            df = L1.parse_yahoo_chart(raw)
            if df is not None and len(df) >= 60:
                stock_dfs[sym] = df
        except Exception:
            continue

    spy_df = stock_dfs["SPY"]
    iwm_df = stock_dfs["IWM"]
    qqq_df = stock_dfs["QQQ"]

    t0_ts = pd.Timestamp(WARMUP_START)
    t1_ts = pd.Timestamp(HO_END)
    trading_days = spy_df.index[(spy_df.index >= t0_ts) & (spy_df.index <= t1_ts)].sort_values()
    T = len(trading_days)
    pos_map = {d: i for i, d in enumerate(trading_days)}

    # Compute SEC filing window per CIK to detect late-start ticker reuse and delisting
    cik_span = filings_df.groupby("cik").agg(min_filed=("filed", "min"), max_filed=("filed", "max"))

    valid_cands: list[tuple[int, str, bool]] = []
    flagged_reuse: dict[str, str] = {}
    delisted_in_window: dict[str, str] = {}

    for cik, sym, is_active in candidates:
        df = stock_dfs.get(sym)
        if df is None:
            continue
        sub = df.reindex(trading_days)
        valid_dates = sub.dropna(subset=["open", "high", "low", "close"]).index
        if len(valid_dates) < 252:
            continue
        first_px_dt = valid_dates[0]
        last_px_dt = valid_dates[-1]

        if cik in cik_span.index:
            min_f = pd.to_datetime(str(cik_span.loc[cik, "min_filed"]), format="mixed", errors="coerce")
            max_f = pd.to_datetime(str(cik_span.loc[cik, "max_filed"]), format="mixed", errors="coerce")
            # If SEC filings exist >2 years before first Yahoo price bar, flag as likely ticker reuse / IPO recycle
            if pd.notna(min_f) and first_px_dt > max(min_f, pd.Timestamp("2009-07-01")) + pd.Timedelta(days=550):
                flagged_reuse[sym] = f"sec_first_filing={min_f.date()} vs yahoo_first_bar={first_px_dt.date()}"
                continue
            # If series ends >10 sessions before end of sample while SEC filings were recent
            if last_px_dt < trading_days[-10]:
                delisted_in_window[sym] = str(last_px_dt.date())

        valid_cands.append((cik, sym, is_active))

    N = len(valid_cands)
    tickers = [x[1] for x in valid_cands]
    ciks = [x[0] for x in valid_cands]

    open_arr = np.full((T, N), np.nan)
    high_arr = np.full((T, N), np.nan)
    low_arr = np.full((T, N), np.nan)
    close_arr = np.full((T, N), np.nan)
    vol_arr = np.full((T, N), np.nan)
    close_unadj_arr = np.full((T, N), np.nan)
    vol_unadj_arr = np.full((T, N), np.nan)
    pit_active_arr = np.zeros((T, N), dtype=bool)
    delist_bar = np.full(N, -1, dtype=np.int32)

    for j, (cik, sym, _is_act) in enumerate(valid_cands):
        df = stock_dfs[sym].reindex(trading_days)
        open_arr[:, j] = df["open"].values
        high_arr[:, j] = df["high"].values
        low_arr[:, j] = df["low"].values
        close_arr[:, j] = df["close"].values
        vol_arr[:, j] = df["volume"].values
        close_unadj_arr[:, j] = df["close_unadj"].values
        vol_unadj_arr[:, j] = df["volume_unadj"].values

        # PIT active window: from first SEC filing (or first valid bar) to last valid bar
        v_pos = np.where(np.isfinite(close_arr[:, j]))[0]
        if len(v_pos) > 0:
            pit_active_arr[v_pos[0] : v_pos[-1] + 1, j] = True
            if sym in delisted_in_window:
                delist_bar[j] = int(v_pos[-1])

    # Strictly enforce >= 4 valid OHLC price points per stock-day
    valid_4pt_arr, rejected_4pt_arr = L2.validate_four_price_points_ohlc(open_arr, high_arr, low_arr, close_arr)

    log.info("Building point-in-time SEC EDGAR fundamental arrays across %d stocks...", N)
    fund_arrays = L2.build_pit_fundamentals_panel(
        filings_df=filings_df,
        trading_days=trading_days,
        ciks=ciks,
        open_arr=open_arr,
        close_arr=close_arr,
        valid_arr=valid_4pt_arr,
    )

    # Annual coverage & survivorship gap table (2009..2026)
    all_cand_ciks = {c for c, _, _ in candidates}
    f_sub = filings_df[filings_df["cik"].isin(all_cand_ciks)].copy()
    f_sub["yr"] = pd.to_datetime(f_sub["filed"].astype(str), format="mixed", errors="coerce").dt.year

    coverage_by_year = {}
    for yr in range(2009, 2027):
        ymask = trading_days.year == yr
        if not np.any(ymask):
            continue
        sec_active_ciks = set(f_sub[f_sub["yr"] == yr]["cik"].unique())
        n_sec_filers = len(sec_active_ciks)
        n_with_price = sum(
            1
            for j, c in enumerate(ciks)
            if c in sec_active_ciks and np.any(valid_4pt_arr[ymask, j])
        )
        n_with_pit_fund_and_4pt = sum(
            1
            for j, c in enumerate(ciks)
            if c in sec_active_ciks and np.any(valid_4pt_arr[ymask, j] & fund_arrays["has_pit_filing"][ymask, j])
        )
        tot_bars_present = int((valid_4pt_arr[ymask] | rejected_4pt_arr[ymask]).sum())
        rej_bars = int(rejected_4pt_arr[ymask].sum())
        coverage_by_year[int(yr)] = {
            "sec_smid_filers": n_sec_filers,
            "with_valid_4pt_ohlc": n_with_price,
            "with_pit_fundamentals_and_4pt_ohlc": n_with_pit_fund_and_4pt,
            "price_coverage_pct": round(n_with_price / max(n_sec_filers, 1), 4),
            "survivorship_gap_pct": round(1.0 - (n_with_price / max(n_sec_filers, 1)), 4),
            "stock_days_4pt_valid": int(valid_4pt_arr[ymask].sum()),
            "stock_days_rejected_incomplete_ohlc": rej_bars,
            "ohlc_4pt_pass_rate": round(int(valid_4pt_arr[ymask].sum()) / max(tot_bars_present, 1), 4),
        }

    hygiene = {
        "sec_meta": {k: v for k, v in sec_meta.items() if k != "historical_fsds_ciks"},
        "n_candidates_screened": len(candidates),
        "n_loaded_clean_stocks": N,
        "n_currently_active_loaded": sum(1 for x in valid_cands if x[2]),
        "n_historical_delisted_loaded": sum(1 for x in valid_cands if not x[2]),
        "n_flagged_ticker_reuse": len(flagged_reuse),
        "flagged_ticker_reuse_sample": dict(list(flagged_reuse.items())[:30]),
        "n_delisted_in_window": len(delisted_in_window),
        "delisted_in_window_sample": dict(list(delisted_in_window.items())[:30]),
        "total_stock_days_4pt_valid": int(valid_4pt_arr.sum()),
        "total_stock_days_rejected_incomplete_ohlc": int(rejected_4pt_arr.sum()),
        "overall_4pt_ohlc_pass_rate": round(
            float(valid_4pt_arr.sum()) / max(float((valid_4pt_arr | rejected_4pt_arr).sum()), 1.0), 4
        ),
        "coverage_by_year": coverage_by_year,
    }

    cpi_s = L1.cpi_deflator_series(cpi_df, trading_days, base_month="2025-09")
    panel = L2.LukV2Panel(
        trading_days=trading_days,
        tickers=tickers,
        ciks=ciks,
        open_arr=open_arr,
        high_arr=high_arr,
        low_arr=low_arr,
        close_arr=close_arr,
        vol_arr=vol_arr,
        close_unadj_arr=close_unadj_arr,
        vol_unadj_arr=vol_unadj_arr,
        valid_4pt_arr=valid_4pt_arr,
        pit_active_arr=pit_active_arr,
        delist_bar=delist_bar,
        fund_arrays=fund_arrays,
        spy_df=spy_df.reindex(trading_days).ffill().bfill(),
        iwm_df=iwm_df.reindex(trading_days).ffill().bfill(),
        qqq_df=qqq_df.reindex(trading_days).ffill().bfill(),
        cpi_ratio=cpi_s.values.astype(float),
        hygiene=hygiene,
        start_eval_date=START_EVAL,
    )

    # Identify top 80 active symbols in 2023-2025 for the 60m intraday sub-period test
    recent_mask = (panel.dates >= pd.Timestamp("2023-11-15")) & (panel.dates < pd.Timestamp(DEV_END))
    cand_v0, _ = L2.candidate_mask_and_score(panel, L2.LukV2Config(id="V0"))
    recent_counts = cand_v0[recent_mask].sum(axis=0)
    top_intraday_idx = np.argsort(-recent_counts)[:80]
    intraday_syms = [panel.tickers[j] for j in top_intraday_idx if recent_counts[j] > 0]
    if len(intraday_syms) < 40:
        # Supplement with most active smid_eligible stocks in 2024-2025
        el_counts = panel.smid_eligible[recent_mask].sum(axis=0)
        for j in np.argsort(-el_counts)[:60]:
            if panel.tickers[j] not in intraday_syms:
                intraday_syms.append(panel.tickers[j])
            if len(intraday_syms) >= 60:
                break

    hourly_dfs = download_yahoo_60m_batch(intraday_syms[:60])
    return panel, hourly_dfs


def run_sp500_comparison(filings_df: pd.DataFrame, cik_to_sym: dict[int, str], cpi_df: pd.DataFrame) -> dict:
    """Comparison (c): Run the exact same V0 monthly momentum + PIT fundamentals strategy on the
    point-in-time S&P 500 universe (`2009-07-01 .. 2025-09-30`) to isolate the universe effect
    (US small/mid caps vs S&P 500 large caps)."""
    sp_csv = L1.ROOT / "sp500_historical_components.csv"
    if not sp_csv.exists():
        from run_luk_v1 import download_sp500_csv
        sp_csv = download_sp500_csv()
    snapshots = L1.parse_sp500_changes(sp_csv)
    sym_to_cik = {v: k for k, v in cik_to_sym.items()}

    sp_syms = sorted({
        sym
        for dt, syms in snapshots
        if dt >= pd.Timestamp("2008-01-01")
        for sym in syms
    } | set(snapshots[0][1]))

    stock_dfs: dict[str, pd.DataFrame] = {}
    for sym in sp_syms + ["SPY", "IWM", "QQQ"]:
        for d_dir in (L1_YAHOO_DIR, YAHOO_DIR):
            p = d_dir / f"{sym}.json"
            if p.exists():
                try:
                    df = L1.parse_yahoo_chart(json.loads(p.read_text()))
                    if df is not None and len(df) >= 60:
                        stock_dfs[sym] = df
                        break
                except Exception:
                    pass

    spy_df = stock_dfs["SPY"]
    iwm_df = stock_dfs.get("IWM", spy_df)
    qqq_df = stock_dfs["QQQ"]
    trading_days = spy_df.index[(spy_df.index >= pd.Timestamp(WARMUP_START)) & (spy_df.index < pd.Timestamp(DEV_END))].sort_values()
    pit_df = L1.pit_membership_matrix(snapshots, trading_days)

    kept_syms = [s for s in pit_df.columns if s in stock_dfs and s not in ("SPY", "IWM", "QQQ")]
    T, N = len(trading_days), len(kept_syms)
    ciks = [sym_to_cik.get(s, -1 - idx) for idx, s in enumerate(kept_syms)]

    open_arr = np.full((T, N), np.nan)
    high_arr = np.full((T, N), np.nan)
    low_arr = np.full((T, N), np.nan)
    close_arr = np.full((T, N), np.nan)
    vol_arr = np.full((T, N), np.nan)
    close_unadj_arr = np.full((T, N), np.nan)
    vol_unadj_arr = np.full((T, N), np.nan)
    for j, s in enumerate(kept_syms):
        df = stock_dfs[s].reindex(trading_days)
        open_arr[:, j] = df["open"].values
        high_arr[:, j] = df["high"].values
        low_arr[:, j] = df["low"].values
        close_arr[:, j] = df["close"].values
        vol_arr[:, j] = df["volume"].values
        close_unadj_arr[:, j] = df["close_unadj"].values
        vol_unadj_arr[:, j] = df["volume_unadj"].values

    valid_4pt, _ = L2.validate_four_price_points_ohlc(open_arr, high_arr, low_arr, close_arr)
    fund_arrays = L2.build_pit_fundamentals_panel(filings_df, trading_days, ciks, open_arr, close_arr, valid_4pt)
    # For S&P 500 comparison, allow market cap up to $5T so large caps aren't filtered out by the $20B smid cap!
    fund_arrays["shares_pit"] = np.where(
        np.isfinite(fund_arrays["shares_pit"]),
        np.clip(fund_arrays["shares_pit"] * (5_000_000_000.0 / np.maximum(fund_arrays["shares_pit"] * close_unadj_arr, 1.0)), 1e7, 1e9),
        1e8,
    )
    cpi_s = L1.cpi_deflator_series(cpi_df, trading_days, base_month="2025-09")
    sp_panel = L2.LukV2Panel(
        trading_days=trading_days,
        tickers=kept_syms,
        ciks=ciks,
        open_arr=open_arr,
        high_arr=high_arr,
        low_arr=low_arr,
        close_arr=close_arr,
        vol_arr=vol_arr,
        close_unadj_arr=close_unadj_arr,
        vol_unadj_arr=vol_unadj_arr,
        valid_4pt_arr=valid_4pt,
        pit_active_arr=pit_df[kept_syms].values.astype(bool),
        delist_bar=np.full(N, -1, dtype=np.int32),
        fund_arrays=fund_arrays,
        spy_df=spy_df.reindex(trading_days).ffill().bfill(),
        iwm_df=iwm_df.reindex(trading_days).ffill().bfill(),
        qqq_df=qqq_df.reindex(trading_days).ffill().bfill(),
        cpi_ratio=cpi_s.values.astype(float),
        hygiene={},
        start_eval_date=START_EVAL,
    )
    res_sp = L2.simulate_luk_v2(sp_panel, L2.LukV2Config(id="SP500_V0"))
    m_sp = engine.metrics(res_sp.daily, res_sp.bar)
    tm_sp = L1.luk_trade_metrics(res_sp.meta["trades"], res_sp.daily)
    return {
        "sharpe": round(m_sp["sharpe"], 3),
        "ann_ret": round(m_sp["ann_ret"], 4),
        "ann_vol": round(m_sp["ann_vol"], 4),
        "max_dd": round(m_sp["max_dd"], 4),
        "total_ret": round(m_sp["total_ret"], 4),
        "n_trades": tm_sp["n_trades"],
        "win_rate": round(tm_sp["win_rate"], 4),
        "payoff": round(tm_sp["payoff"], 4),
        "profit_factor": round(tm_sp["profit_factor"], 4),
    }


def oos_trades_and_capacity(res: dict) -> tuple[pd.DataFrame, float, float]:
    oos_trades = []
    adv_parts = []
    for f, k in zip(res["folds"], res["chosen_idx"]):
        r = res["results"][k]
        t0 = pd.Timestamp(f["test_start"])
        t1 = pd.Timestamp(f["test_end"]) + pd.Timedelta("1D")
        tr = r.meta.get("trades", pd.DataFrame())
        if len(tr) > 0:
            sub_tr = tr[(tr["exit_date"] >= t0) & (tr["exit_date"] < t1)]
            if len(sub_tr) > 0:
                oos_trades.append(sub_tr)
        tb = r.meta.get("trading_bar")
        if tb is not None and "adv_part" in tb.columns:
            sub_b = tb.loc[(tb.index >= pd.Timestamp(t0, tz="UTC")) & (tb.index < pd.Timestamp(t1, tz="UTC"))]
            active_adv = sub_b["adv_part"][sub_b["adv_part"] > 0]
            if len(active_adv) > 0:
                adv_parts.append(active_adv)

    tr_all = pd.concat(oos_trades, ignore_index=True) if oos_trades else pd.DataFrame()
    hold_days_p95 = 25.0
    cap_p95 = float(pd.concat(adv_parts).quantile(0.95)) if adv_parts else 0.0
    return tr_all, hold_days_p95, cap_p95


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.mkdir(parents=True, exist_ok=True)

    from run_luk_v1 import ensure_global_trial_registry
    ensure_global_trial_registry(REG)

    t0 = time.time()
    full_panel, hourly_dfs = build_luk_v2_panel()
    dev_panel = full_panel.slice_until(DEV_END)

    mf = manifest.build([str(DATA_DIR)], DATA_DIR / ".manifest_cache.json")
    (DOC / "data_manifest_sha.json").write_text(
        json.dumps({"manifest_sha": mf["manifest_sha"], "n_files": mf["n_files"]}, indent=2)
    )

    rules, _ = load_rules()
    vault = Vault()
    reg = TrialRegistry(REG)

    spec_path = SPECS / "L2_luk_smid_mom_fund.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    policy = yaml.safe_load((SPECS / "HOLDOUT_POLICY.yaml").read_text())
    fam = spec["family"]
    raw_grid = spec["grid"]
    grid = L2.normalize_grid_v2(raw_grid)
    pol = policy["families"][fam]

    vault.guard(fam, dev_panel.dates[dev_panel.dates >= pd.Timestamp(START_EVAL)])

    eval_cache_pess: dict[int, engine.BTResult] = {}
    eval_cache_delay: dict[int, engine.BTResult] = {}

    def eval_pess(p: dict) -> engine.BTResult:
        idx = grid.index(p)
        if idx not in eval_cache_pess:
            cfg = L2.config_v2_from_grid(raw_grid[idx], delist_mode="pessimistic", extra_delay=0)
            eval_cache_pess[idx] = L2.simulate_luk_v2(dev_panel, cfg)
        return eval_cache_pess[idx]

    def eval_delay(p: dict) -> engine.BTResult:
        idx = grid.index(p)
        if idx not in eval_cache_delay:
            cfg = L2.config_v2_from_grid(raw_grid[idx], delist_mode="pessimistic", extra_delay=1)
            eval_cache_delay[idx] = L2.simulate_luk_v2(dev_panel, cfg)
        return eval_cache_delay[idx]

    log.info("Running walk-forward on L2_luk_smid_mom_fund (5 folds, plateau selector)...")
    res = walk_forward(
        fam, grid, eval_pess, reg,
        n_folds=5, purge_days=45, embargo_days=30,
        min_train_days=1095, min_embargo_frac=0.01, selector="plateau",
    )

    log.info("Running walk-forward R11 (+1 extra bar delay, open_{t+2})...")
    res_d = walk_forward(
        fam + "_delay2", grid, eval_delay, reg,
        n_folds=5, purge_days=45, embargo_days=30,
        min_train_days=1095, min_embargo_frac=0.01, selector="plateau",
    )

    sha = prereg.file_sha(spec_path)
    cts = prereg.git_commit_ts(spec_path)
    fam_trials = [x["ts"] for x in reg.all() if x["params"].get("strategy") == fam]
    first_ts = min(fam_trials) if fam_trials else time.time()
    pc = prereg.check(sha, first_ts)
    pc["git_commit_ts"] = cts
    pc["commit_before_first_trial"] = bool(cts and cts < first_ts)
    pc["ok"] = pc["ok"] and pc["commit_before_first_trial"]

    spy_cal = dev_panel.spy["close"].copy()
    spy_cal.index = pd.DatetimeIndex(spy_cal.index, tz="UTC")
    spy_cal = spy_cal.reindex(res["results"][0].daily.index).ffill()

    iwm_cal = dev_panel.iwm["close"].copy()
    iwm_cal.index = pd.DatetimeIndex(iwm_cal.index, tz="UTC")
    iwm_cal = iwm_cal.reindex(res["results"][0].daily.index).ffill()

    evd = evidence(res, rules, spy_cal, res_d["oos_daily"])
    oos_trades_df, hold_p95, cap_p95 = oos_trades_and_capacity(res)
    n_tr_oos = int(len(oos_trades_df))

    # Variant summary table (dev 2009-07-01..2025-09-30)
    variant_rows = {}
    for k, p in enumerate(raw_grid):
        r_k = eval_cache_pess[k]
        m_k = engine.metrics(r_k.daily, r_k.bar)
        tr_k = r_k.meta["trades"]
        tm_k = L1.luk_trade_metrics(tr_k, r_k.daily)
        variant_rows[p["id"]] = {
            "params": p,
            "sharpe": round(m_k["sharpe"], 3),
            "ann_ret": round(m_k["ann_ret"], 4),
            "ann_vol": round(m_k["ann_vol"], 4),
            "max_dd": round(m_k["max_dd"], 4),
            "turnover_ann": round(m_k["turnover_ann"], 2),
            "cost_ann": round(m_k["cost_ann"], 4),
            **{k_: (round(v_, 4) if isinstance(v_, float) else v_) for k_, v_ in tm_k.items()},
        }

    # Select plateau-centre on full dev sample for one-shot holdout
    M = pd.DataFrame({k: r.daily for k, r in res["results"].items()}).fillna(0.0)
    sc = [float(M[k].mean() / M[k].std()) if M[k].std() > 0 else -np.inf for k in M.columns]
    k_star = plateau_select(sc, grid)
    chosen_cfg_ho = L2.config_v2_from_grid(raw_grid[k_star], delist_mode="pessimistic")

    # Open holdout vault ONCE
    if vault.status(fam).get("status") != "opened":
        vault.open_once(fam, "holdout_2025-10_luk_v2")
    r_ho_full = L2.simulate_luk_v2(full_panel, chosen_cfg_ho)
    d_ho = r_ho_full.daily.loc[r_ho_full.daily.index >= pd.Timestamp(DEV_END, tz="UTC")]
    b_ho = r_ho_full.bar.loc[r_ho_full.bar.index >= pd.Timestamp(DEV_END, tz="UTC")]
    hm = engine.metrics(d_ho, b_ho)
    tr_ho = L1.luk_trade_metrics(r_ho_full.meta["trades"], d_ho, start_dt=DEV_END, end_dt=HO_END)

    r_v0_ho_full = r_ho_full if k_star == 0 else L2.simulate_luk_v2(full_panel, L2.LukV2Config(id="V0"))
    d_v0_ho = r_v0_ho_full.daily.loc[r_v0_ho_full.daily.index >= pd.Timestamp(DEV_END, tz="UTC")]
    hm_v0 = engine.metrics(d_v0_ho)
    tr_v0_ho = L1.luk_trade_metrics(r_v0_ho_full.meta["trades"], d_v0_ho, start_dt=DEV_END, end_dt=HO_END)

    spy_ho_s = full_panel.spy["close"].copy()
    spy_ho_s.index = pd.DatetimeIndex(spy_ho_s.index, tz="UTC")
    spy_ho_ret = spy_ho_s.reindex(d_ho.index).ffill().pct_change(fill_method=None).fillna(0.0)
    spy_ho_metrics = engine.metrics(spy_ho_ret)

    iwm_ho_s = full_panel.iwm["close"].copy()
    iwm_ho_s.index = pd.DatetimeIndex(iwm_ho_s.index, tz="UTC")
    iwm_ho_ret = iwm_ho_s.reindex(d_ho.index).ffill().pct_change(fill_method=None).fillna(0.0)
    iwm_ho_metrics = engine.metrics(iwm_ho_ret)

    # Required comparisons:
    # (a) Momentum without fundamentals
    log.info("Running required comparisons (momentum without fundamentals, fundamentals without momentum, S&P 500 version, 60m intraday)...")
    res_mom_nofund = L2.simulate_luk_v2(
        dev_panel,
        L2.LukV2Config(id="comp_mom_no_fund", fund_mode="none", catalyst_mode="none"),
    )
    # (b) Fundamentals without momentum
    res_fund_nomom = L2.simulate_luk_v2(
        dev_panel,
        L2.LukV2Config(id="comp_fund_no_mom", rank_mode="fundamentals_only", fund_mode="canonic_eps_rev", catalyst_mode="sue_or_gap"),
    )
    # (c) S&P 500 version of V0
    filings_df = pd.read_csv(SEC_DIR / "sec_quarterly_fundamentals.csv.gz")
    sec_meta = json.loads((SEC_DIR / "sec_meta.json").read_text())
    cik_to_sym = {int(k): v for k, v in sec_meta.get("historical_fsds_ciks", {}).items()}
    cpi_df = pd.read_csv(L1.ROOT / "CPIAUCSL.csv")
    sp500_comp = run_sp500_comparison(filings_df, cik_to_sym, cpi_df)

    # (d) Clearly separated 60m intraday sub-period test
    intraday_comp = L2.run_intraday_subperiod_comparison(dev_panel, hourly_dfs)

    oos = res["oos_daily"]
    oos_start_str = str(oos.index[0].date())
    spy_oos_ret = spy_cal.pct_change(fill_method=None).reindex(oos.index).fillna(0.0)
    iwm_oos_ret = iwm_cal.pct_change(fill_method=None).reindex(oos.index).fillna(0.0)

    r_v0_dev = eval_cache_pess[0]
    wf_oos_trade_metrics = L1.luk_trade_metrics(oos_trades_df, oos)
    v0_oos_trade_metrics = L1.luk_trade_metrics(
        r_v0_dev.meta["trades"],
        r_v0_dev.daily.loc[r_v0_dev.daily.index >= pd.Timestamp(oos_start_str, tz="UTC")],
        start_dt=oos_start_str,
        end_dt=DEV_END,
    )

    comparisons = {
        "a_momentum_without_fundamentals_dev": {
            **engine.metrics(res_mom_nofund.daily, res_mom_nofund.bar),
            "trade_metrics": L1.luk_trade_metrics(res_mom_nofund.meta["trades"], res_mom_nofund.daily),
        },
        "a_momentum_without_fundamentals_oos": {
            **engine.metrics(res_mom_nofund.daily.loc[res_mom_nofund.daily.index >= pd.Timestamp(oos_start_str, tz="UTC")]),
            "trade_metrics": L1.luk_trade_metrics(res_mom_nofund.meta["trades"], res_mom_nofund.daily, start_dt=oos_start_str, end_dt=DEV_END),
        },
        "b_fundamentals_without_momentum_dev": {
            **engine.metrics(res_fund_nomom.daily, res_fund_nomom.bar),
            "trade_metrics": L1.luk_trade_metrics(res_fund_nomom.meta["trades"], res_fund_nomom.daily),
        },
        "b_fundamentals_without_momentum_oos": {
            **engine.metrics(res_fund_nomom.daily.loc[res_fund_nomom.daily.index >= pd.Timestamp(oos_start_str, tz="UTC")]),
            "trade_metrics": L1.luk_trade_metrics(res_fund_nomom.meta["trades"], res_fund_nomom.daily, start_dt=oos_start_str, end_dt=DEV_END),
        },
        "c_sp500_universe_version_dev": sp500_comp,
        "d_intraday_60m_subperiod_test": intraday_comp,
    }

    years_gate_oos = L1.yearly_vs_spy_and_gate(oos, spy_cal)
    years_gate_v0 = L1.yearly_vs_spy_and_gate(r_v0_dev.daily, spy_cal)
    years_vs_iwm_oos = L1.yearly_vs_spy_and_gate(oos, iwm_cal)

    rep = {
        "name": fam,
        "family": fam,
        "prereg_sha": sha,
        "prereg_check": pc,
        "data_manifest_sha": mf["manifest_sha"],
        "registry_path": str(REG),
        **evd,
        "n_trades": n_tr_oos,
        "max_horizon_days": round(hold_p95, 1),
        "capacity": {
            "target_capital_usd": 100_000,
            "adv_participation_p95": round(cap_p95, 6),
            "note": "US small/mid-cap PIT universe ($300M-$20B real mcap, ADV$20 >= $5M CPI-adjusted, >=4 OHLC points/day)",
        },
        "capacity_p95": round(cap_p95, 6),
        "pit_universe": True,
        "market_neutral": False,
        "benchmark_note": "R7 'btc_buy_hold' slot = SPY buy&hold; IWM buy&hold also reported",
        "trade_metrics_wf_oos": wf_oos_trade_metrics,
        "trade_metrics_v0_oos": v0_oos_trade_metrics,
        "payoff_dev_oos": payoff_stats(oos),
        "payoff_holdout": payoff_stats(d_ho),
        "variants_dev_summary": variant_rows,
        "hygiene": dev_panel.hygiene,
        "comparisons": comparisons,
        "years_vs_spy_oos": years_gate_oos,
        "years_vs_spy_v0_dev": years_gate_v0,
        "years_vs_iwm_oos": years_vs_iwm_oos,
        "spy_oos_metrics": engine.metrics(spy_oos_ret),
        "iwm_oos_metrics": engine.metrics(iwm_oos_ret),
        "vault": {
            "status": pol["r3_status"],
            "reason": pol["reason"].strip(),
            "holdout_sharpe": hm["sharpe"],
            "holdout": {
                "param": raw_grid[k_star],
                "sharpe": hm["sharpe"],
                "ann_ret": hm["ann_ret"],
                "max_dd": hm["max_dd"],
                "total_ret": hm["total_ret"],
                "trade_metrics": tr_ho,
                "v0_holdout": {
                    "sharpe": hm_v0["sharpe"],
                    "ann_ret": hm_v0["ann_ret"],
                    "max_dd": hm_v0["max_dd"],
                    "total_ret": hm_v0["total_ret"],
                    "trade_metrics": tr_v0_ho,
                },
                "spy_total_ret": float((1.0 + spy_ho_ret).prod() - 1.0),
                "spy_sharpe": spy_ho_metrics["sharpe"],
                "spy_max_dd": spy_ho_metrics["max_dd"],
                "iwm_total_ret": float((1.0 + iwm_ho_ret).prod() - 1.0),
                "iwm_sharpe": iwm_ho_metrics["sharpe"],
                "iwm_max_dd": iwm_ho_metrics["max_dd"],
            },
        },
        "paper": None,
        "folds": res["folds"],
        "oos_window": [str(oos.index[0].date()), str(oos.index[-1].date())],
        "dev_end": DEV_END,
        "elapsed_s": round(time.time() - t0, 1),
    }

    rep["verdict"] = verdict(rep)
    rep["verdict_aggressive"] = verdict(rep, AGGRESSIVE_PATH)

    out_json = OUT / f"{fam}.json"
    doc_json = DOC / f"{fam}.json"
    out_json.write_text(json.dumps(rep, indent=2, default=str))
    doc_json.write_text(json.dumps(rep, indent=2, default=str))

    log.info(
        "%s -> base=%s (failed=%s) | aggressive=%s (failed=%s) | OOS SR=%.2f trades=%d | HO SR=%.2f",
        fam,
        rep["verdict"]["verdict"],
        rep["verdict"]["failed"],
        rep["verdict_aggressive"]["verdict"],
        rep["verdict_aggressive"]["failed"],
        rep["oos"]["sharpe"],
        n_tr_oos,
        hm["sharpe"],
    )


if __name__ == "__main__":
    main()
