"""Build the pass/fail table (markdown + csv) from reports/research_rules/*.json and copy the
evidence to docs/research/rules_rescore_2026-10/ (manifest sha only; full manifest gzipped)."""
import gzip
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "reports/research_rules", ROOT / "docs/research/rules_rescore_2026-10"
ORDER = ["S1_basis_arb", "S2_funding_carry", "S3_donchian_ensemble", "S4_tsmom_atr_6h", "S5_leadlag_1m",
         "S6_pairs_coint", "C1_intraday_tsmom_30m", "C2_donchian_15m", "C3_squeeze_4h", "C4_funding_extremes",
         "C5_cascade_1h", "C6_alt_tsmom_weekly_pit", "C7_S3_donchian_pit20", "C8_new_listings"]


def main():
    DST.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in ORDER:
        f = SRC / f"{n}.json"
        if not f.exists():
            rows.append({"strategy": n, "verdict": "NOT RUN"})
            continue
        r = json.loads(f.read_text())
        shutil.copy(f, DST / f.name)
        row = {"strategy": n, "oos_sharpe": round(r["oos"]["sharpe"], 2)}
        for k, c in r["verdict"]["checks"].items():
            row[k] = "✅" if c["pass"] else "❌"
        row["verdict"] = r["verdict"]["verdict"]
        row["failed"] = ",".join(r["verdict"]["failed"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(DST / "table.csv", index=False)
    mf = json.loads((SRC / "data_manifest.json").read_text())
    (DST / "data_manifest_sha.json").write_text(json.dumps({"manifest_sha": mf["manifest_sha"],
                                                            "n_files": mf["n_files"]}))
    with gzip.open(DST / "data_manifest.json.gz", "wt") as fh:
        json.dump(mf, fh)
    t = df.drop(columns=["failed"]).astype(str)
    md = ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    md += ["| " + " | ".join(r) + " |" for r in t.values]
    (DST / "table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(main())
