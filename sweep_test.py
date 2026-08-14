"""Threshold/size-floor sweep against known-good scans.

    python sweep_test.py [n_random_from_D]

Two references, because two things must both hold:
  * healthy scans  -> current Otsu count (known good); a method passes within 10%
  * degenerate scans -> hand counts (Pot_200 = 140, Pot_208 = 12)

A method only passes if it clears BOTH. Writes SweepResults.csv.
Reads images only; changes nothing.
"""
import os, re, sys, gc, random
import numpy as np
import pandas as pd
import imageio.v3 as iio
from skimage.filters import threshold_otsu, threshold_triangle, threshold_yen
from skimage.measure import label, regionprops_table
from skimage.morphology import remove_small_objects
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
pd.set_option("display.width", 260)

PP_SQMM = (1200 / 25.4) ** 2
DL = os.path.join(os.environ["USERPROFILE"], "Downloads", "drive-download-20260812T192909Z-1-001")
DALL = r"D:\ALL"
HERE = os.path.dirname(os.path.abspath(__file__))

HAND = {"Pot_200": 140, "Pot_208": 12}          # counted by hand
FLOORS = [0.60, 0.35, 0.25, 0.20, 0.15]
SEED_MAX, MAX_AR, MIN_SOL = 4.0, 5.5, 0.45
DEGENERATE_FG = 0.15        # >15% of a scan being "seed" is physically impossible here


def scan_id(fn):
    m = re.search(r"\{Pot_(\d+)\}", fn, re.I)
    if m:
        return f"Pot_{m.group(1)}"
    m = re.search(r"\{Plot#?\d*_(\d+)\}", fn, re.I)
    if m:
        return f"Plot_{m.group(1)}"
    m = re.search(r"\{Order_(\d+)\}", fn, re.I)
    return f"Order_{m.group(1)}" if m else fn[:24]


def regions(gray, thresh):
    """Region table at one threshold. Size floors are applied afterwards, for free."""
    binary = remove_small_objects(gray > thresh, min_size=int(PP_SQMM * 0.05))
    fg = float(binary.mean())
    df = pd.DataFrame(regionprops_table(label(binary),
         properties=["area", "solidity", "major_axis_length", "minor_axis_length"]))
    if df.empty:
        return df, fg
    df["mm2"] = df["area"] / PP_SQMM
    df["ar"] = df["major_axis_length"] / df["minor_axis_length"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df, fg


def count_at(df, floor):
    if df.empty:
        return 0
    return int(((df["mm2"] >= floor) & (df["mm2"] <= SEED_MAX)
                & (df["ar"] <= MAX_AR) & (df["solidity"] >= MIN_SOL)).sum())


def main(n_random=35):
    files = [os.path.join(DL, f) for f in sorted(os.listdir(DL))]
    pool = [os.path.join(DALL, f) for f in os.listdir(DALL) if f.lower().endswith((".tif", ".tiff"))]
    random.Random(7).shuffle(pool)
    files += pool[:n_random]
    print(f"{len(files)} scans: {len(files)-n_random} from Downloads, {n_random} random from D:\\ALL\n")

    rows = []
    for i, path in enumerate(files, 1):
        sid = scan_id(os.path.basename(path))
        g = iio.imread(path)
        gray = g.astype(np.float32) if g.ndim == 2 else np.mean(g[:, :, :3], axis=2).astype(np.float32)
        del g; gc.collect()
        sub = gray[::4, ::4]
        med = float(np.median(sub))
        mad = float(np.median(np.abs(sub - med))) * 1.4826

        cand = {"otsu": float(threshold_otsu(sub)),
                "triangle": float(threshold_triangle(sub)),
                "yen": float(threshold_yen(sub)),
                "fixed_62": 62.0,
                "bg+6mad": med + 6 * mad,
                "bg+8mad": med + 8 * mad}

        per_thresh = {}
        for name, t in cand.items():
            df, fg = regions(gray, t)
            per_thresh[name] = (df, fg)
            for fl in FLOORS:
                rows.append({"scan": sid, "method": name, "floor": fl,
                             "thresh": round(t, 1), "fg": round(fg, 4), "count": count_at(df, fl)})

        # hybrid: trust Otsu unless it produced a physically impossible foreground, then fall back
        otsu_df, otsu_fg = per_thresh["otsu"]
        for fb in ("fixed_62", "bg+8mad"):
            df, fg = (otsu_df, otsu_fg) if otsu_fg <= DEGENERATE_FG else per_thresh[fb]
            t = cand["otsu"] if otsu_fg <= DEGENERATE_FG else cand[fb]
            for fl in FLOORS:
                rows.append({"scan": sid, "method": f"guarded_otsu->{fb}", "floor": fl,
                             "thresh": round(t, 1), "fg": round(fg, 4), "count": count_at(df, fl)})

        print(f"  [{i}/{len(files)}] {sid:14s} otsu={cand['otsu']:5.1f} fg={otsu_fg*100:5.2f}%"
              f"{'  <-- OTSU DEGENERATE' if otsu_fg > DEGENERATE_FG else ''}", flush=True)
        del gray, per_thresh; gc.collect()

    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "SweepResults.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    summarise(df)


def summarise(df):
    # reference per scan: hand count if we have one, else that scan's current Otsu @ floor 0.60
    base = df[(df["method"] == "otsu") & (df["floor"] == 0.60)].set_index("scan")
    degenerate = set(base.index[base["fg"] > DEGENERATE_FG])
    ref, kind = {}, {}
    for sid in base.index:
        if sid in HAND:
            ref[sid], kind[sid] = HAND[sid], "hand-counted"
        elif sid in degenerate:
            kind[sid] = "degenerate (no reference)"
        else:
            ref[sid], kind[sid] = int(base.loc[sid, "count"]), "healthy (otsu ref)"

    print(f"\nscans: {len(base)}   degenerate under current Otsu: {len(degenerate)}")
    print(f"  {sorted(degenerate)}\n")

    df = df[df["scan"].isin(ref)].copy()
    df["ref"] = df["scan"].map(ref)
    df["kind"] = df["scan"].map(kind)
    df["ratio"] = df["count"] / df["ref"]
    df["within10"] = df["ratio"].between(0.9, 1.1)

    g = df.groupby(["method", "floor"])
    res = pd.DataFrame({
        "n": g.size(),
        "pass_within_10pct": g["within10"].sum(),
        "worst_ratio": g["ratio"].apply(lambda s: max(s.max(), 1 / s.min()) if s.min() > 0 else np.inf),
    })
    res["pct_pass"] = (100 * res["pass_within_10pct"] / res["n"]).round(1)
    res = res.sort_values(["pct_pass", "worst_ratio"], ascending=[False, True])
    print("===== every method x floor, ranked =====")
    print(res.to_string())

    hand = df[df["kind"] == "hand-counted"]
    if not hand.empty:
        print("\n===== on the two hand-counted scans =====")
        print(hand.pivot_table(index=["method", "floor"], columns="scan",
                               values="count").to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 35)
