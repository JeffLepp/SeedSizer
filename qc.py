"""Reconcile SeedSizer counts against balance weights.

    python qc.py results.xlsx [-o flagged.xlsx]

The weight comes off a balance and is trustworthy. The count is inferred from an image
and is not. So thousand-seed mass (TSM = weight / count * 1000) is a physical check on
the count: a batch whose TSM is far off the rest of the run has a wrong count, and the
weight tells you what the count should have been.

The expected TSM is the run's own median, so this works for any crop without tuning.
"""
import argparse
import sys

import numpy as np
import pandas as pd

TOLERANCE = 2.0     # flag when TSM is >2x off the run median (either direction)
MIN_COUNT = 50      # below this there is too little seed for a count to mean anything


def _pick(df, *names):
    lookup = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n in lookup:
            return lookup[n]
    return None


def check(df, count_col, weight_col, tolerance=TOLERANCE, expected_tsm=None):
    """Add TSM, ImpliedCount, Status and Reason columns. Returns a new frame."""
    out = df.copy()
    count = pd.to_numeric(out[count_col], errors="coerce")
    weight = pd.to_numeric(out[weight_col], errors="coerce")

    tsm = np.where(count > 0, weight / count * 1000, np.nan)
    out["TSM"] = tsm

    if expected_tsm is None:
        expected_tsm = np.nanmedian(tsm)
    if not np.isfinite(expected_tsm) or expected_tsm <= 0:
        raise SystemExit("Could not establish an expected TSM; check the weight and count columns.")

    # what the count should have been, given the weight
    implied = weight / expected_tsm * 1000
    out["ImpliedCount"] = implied.round()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(count > 0, implied / count, np.nan)
    out["CountRatio"] = ratio                      # <1 means SeedSizer over-counted

    status, reason = [], []
    for c, w, t, imp, r in zip(count, weight, tsm, implied, ratio):
        if not np.isfinite(c) or not np.isfinite(w):
            status.append("FAIL"); reason.append("Missing count or weight."); continue
        if c <= 0:
            status.append("FAIL"); reason.append("No seeds detected."); continue
        if np.isfinite(imp) and imp < MIN_COUNT:
            status.append("FAIL")
            reason.append(f"Weight implies only ~{imp:.0f} seeds; too little seed to count from a scan.")
            continue
        if not np.isfinite(r) or r > tolerance:
            status.append("FAIL"); reason.append(f"TSM {t:.3f} far below run median {expected_tsm:.3f}; count is too high.")
        elif r < 1 / tolerance:
            status.append("FAIL"); reason.append(f"TSM {t:.3f} far above run median {expected_tsm:.3f}; count is too low.")
        elif r > 1.25 or r < 0.8:
            status.append("CHECK"); reason.append(f"TSM {t:.3f} drifts from run median {expected_tsm:.3f}.")
        else:
            status.append("OK"); reason.append("")

    out["Status"] = status
    out["Reason"] = reason
    out.attrs["expected_tsm"] = expected_tsm
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help=".xlsx or .csv of SeedSizer results with a weight column")
    ap.add_argument("-o", "--out", help="write the annotated table here (.xlsx or .csv)")
    ap.add_argument("--count", help="count column name (default: auto-detect)")
    ap.add_argument("--weight", help="weight column name, in grams (default: auto-detect)")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE, help=f"fold-change before FAIL (default {TOLERANCE})")
    ap.add_argument("--expected-tsm", type=float, help="known TSM for the crop; default is the run median")
    a = ap.parse_args(argv)

    df = pd.read_excel(a.path) if a.path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(a.path)

    count_col = a.count or _pick(df, "sscount", "seed_count", "count")
    weight_col = a.weight or _pick(df, "weight.g", "weight_g", "weight")
    if not count_col or not weight_col:
        raise SystemExit(f"Need a count and a weight column. Found: {list(df.columns)}")

    res = check(df, count_col, weight_col, a.tolerance, a.expected_tsm)
    id_col = _pick(res, "packet", "potid", "filename", "file") or res.columns[0]

    counts = res["Status"].value_counts()
    print(f"{a.path}")
    print(f"  rows {len(res)}   expected TSM {res.attrs['expected_tsm']:.3f} g/1000 seeds")
    for s in ("OK", "CHECK", "FAIL"):
        print(f"  {s:<6} {counts.get(s, 0)}")

    bad = res[res["Status"] == "FAIL"]
    if not bad.empty:
        print(f"\n  worst offenders:")
        cols = [id_col, count_col, weight_col, "TSM", "ImpliedCount", "Reason"]
        show = bad.nsmallest(min(10, len(bad)), "TSM")[cols].copy()
        show[id_col] = show[id_col].astype(str).str.slice(0, 46)
        print(show.to_string(index=False))

    if a.out:
        res.to_excel(a.out, index=False) if a.out.lower().endswith((".xlsx", ".xls")) else res.to_csv(a.out, index=False)
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
