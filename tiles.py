"""Hand-verification by tiles, instead of counting a whole scan.

    python tiles.py sample <scan.tif> [n_tiles] [tile_mm]
    python tiles.py score  <tiles_dir>

Counting 1,149 seeds off one scan is not realistic. Counting 30 seeds in a
25 mm window is, and it tests the same pixels through the same code. `sample`
cuts windows out of a real scan, records what SeedSizer found inside each one,
and writes an HTML page with the images so a count can be entered by hand.

Tiles are chosen on purpose, not at random: they are binned by how much seed
SeedSizer thinks is in them and sampled across every bin, so empty bed, sparse
bed and heavy clumping all end up in the set. Random sampling on a mostly-empty
A4 bed returns mostly empty tiles.

IMPORTANT: detection runs on the WHOLE scan first, exactly as in production, and
objects are then assigned to tiles by centroid. Running detection on a tile in
isolation would compute its threshold and reference seed area from that tile
alone, which is not what happens in production and would measure the wrong
thing. The tile counts are checked to sum to SeedSizer.Run()'s own total.
"""
import os, sys, csv, html, io, contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import SeedSizer as S
from skimage.measure import label, regionprops_table
from skimage.morphology import remove_small_objects
import pandas as pd
import imageio.v3 as iio
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
PX_PER_MM = S.PPI / 25.4


def detect(path):
    """Reproduce SeedSizer's own object table, with per-object seed counts.

    Deliberately mirrors Run(); the caller asserts the totals agree, so if
    SeedSizer changes and this drifts, the run fails instead of quietly lying.
    """
    raw = iio.imread(path)
    gray = S._as_grayscale_float(raw)
    del raw

    t = S.threshold_otsu(gray)
    degenerate = float((gray > t).mean()) > S.DEGENERATE_FG
    if degenerate:
        t = S.FALLBACK_THRESHOLD
    binary = remove_small_objects(gray > t, min_size=int(S.PP_SQMM * S.FILTER))
    df = pd.DataFrame(regionprops_table(label(binary), properties=[
        "label", "area", "centroid", "eccentricity", "solidity",
        "major_axis_length", "minor_axis_length"]))
    if df.empty:
        return gray, df, t, degenerate

    df["area_mm2"] = df["area"] / S.PP_SQMM
    df["aspect_ratio"] = df["major_axis_length"] / df["minor_axis_length"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    plausible = df[(df["area_mm2"].between(S.MIN_SEED_AREA_MM2, S.MAX_SINGLE_SEED_AREA_MM2))
                   & (df["aspect_ratio"] <= S.MAX_SINGLE_ASPECT_RATIO)
                   & (df["solidity"] >= S.MIN_SOLIDITY)]
    src = df if plausible.empty else plausible
    ref = max(float(src["area_mm2"].median()), S.MIN_REFERENCE_SEED_AREA_MM2)

    min_area = max(S.MIN_SEED_AREA_MM2, S.MINSIZE * ref)
    shape_ok = (df["aspect_ratio"] <= S.MAX_SINGLE_ASPECT_RATIO) & (df["solidity"] >= S.MIN_SOLIDITY)
    clump_ok = ((df["aspect_ratio"] <= S.MAX_CLUMP_ASPECT_RATIO)
                & (df["solidity"] >= S.MIN_SOLIDITY * 0.65)
                & (df["area_mm2"] <= S.MAX_CLUMP_AREA_MM2))
    keep = df[(df["area_mm2"] >= min_area)
              & (shape_ok | ((df["area_mm2"] > S.CLUMP_FACTOR * ref) & clump_ok))].copy()

    is_clump = keep["area_mm2"] > S.CLUMP_FACTOR * ref
    keep["seeds"] = 1
    keep.loc[is_clump, "seeds"] = np.maximum(2, (keep.loc[is_clump, "area_mm2"] / ref).round().astype(int))
    keep["is_clump"] = is_clump
    return gray, keep, t, degenerate


def stretch(a):
    lo, hi = np.percentile(a, 2), np.percentile(a, 99.9)
    return (255 * np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)).astype(np.uint8)


# Seeds sliced by a tile edge are the one thing a human cannot count reliably:
# half a seed is neither clearly in nor clearly out. Each tile is therefore saved
# with a margin of surrounding scan and the true boundary drawn on it, so whole
# seeds are always visible and the rule is simply "count the ones whose CENTRE is
# inside the line" - which is exactly how objects are assigned to tiles here.
TILE_MARGIN_MM = 4.0


def sample(scan, n_tiles=24, tile_mm=50.0):
    n_tiles, tile_mm = int(n_tiles), float(tile_mm)
    out = os.path.join(HERE, "TileCheck", os.path.splitext(os.path.basename(scan))[0][:40])
    os.makedirs(out, exist_ok=True)

    print(f"detecting on the whole scan ({os.path.basename(scan)}) ...", flush=True)
    gray, keep, thr, degenerate = detect(scan)
    with contextlib.redirect_stdout(io.StringIO()):
        official = int(S.Run(scan)["sscount"])
    total = int(keep["seeds"].sum()) if len(keep) else 0
    if total != official:
        sys.exit(f"tile detection ({total}) disagrees with SeedSizer.Run() ({official}); "
                 f"tiles.py has drifted from SeedSizer.py and must be updated.")
    print(f"  threshold={thr:.1f}{'  [degenerate -> fallback]' if degenerate else ''}   "
          f"total seeds={total}", flush=True)

    tp = int(round(tile_mm * PX_PER_MM))
    H, W = gray.shape
    ny, nx = H // tp, W // tp
    cy = keep["centroid-0"].to_numpy() if len(keep) else np.zeros(0)
    cx = keep["centroid-1"].to_numpy() if len(keep) else np.zeros(0)

    cells = []
    for iy in range(ny):
        for ix in range(nx):
            y0, x0 = iy * tp, ix * tp
            m = (cy >= y0) & (cy < y0 + tp) & (cx >= x0) & (cx < x0 + tp)
            sub = keep[m] if len(keep) else keep
            cells.append({"iy": iy, "ix": ix, "y0": y0, "x0": x0,
                          "pred": int(sub["seeds"].sum()) if len(sub) else 0,
                          "objects": int(len(sub)),
                          "clumps": int(sub["is_clump"].sum()) if len(sub) else 0})

    # Bin by predicted density and take from every bin, so empty bed, sparse bed
    # and heavy clumping are all represented rather than whatever chance gives.
    ak = (tile_mm / 25.0) ** 2                     # bins scale with tile area
    BINS = [(0, 0, "empty"), (1, int(5 * ak), "very sparse"),
            (int(5 * ak) + 1, int(20 * ak), "sparse"),
            (int(20 * ak) + 1, int(50 * ak), "moderate"),
            (int(50 * ak) + 1, 10 ** 9, "dense")]
    picked, rng = [], np.random.default_rng(0)
    per = max(1, n_tiles // len(BINS))
    for lo, hi, name in BINS:
        pool = [c for c in cells if lo <= c["pred"] <= hi]
        if not pool:
            print(f"  bin {name:12s}: none on this scan")
            continue
        clumpy = sorted(pool, key=lambda c: -c["clumps"])[:per]        # prefer clumped
        rest = [c for c in pool if c not in clumpy]
        extra = list(rng.choice(rest, size=min(per - len(clumpy), len(rest)), replace=False)) if rest else []
        take = (clumpy + extra)[:per]
        for c in take:
            c["bin"] = name
        picked += take
        print(f"  bin {name:12s}: {len(pool):5d} tiles available, took {len(take)}")

    mg = int(round(TILE_MARGIN_MM * PX_PER_MM))
    rows = []
    for k, c in enumerate(sorted(picked, key=lambda c: c["pred"])):
        y0, x0 = c["y0"], c["x0"]
        ya, yb = max(y0 - mg, 0), min(y0 + tp + mg, H)   # crop with context
        xa, xb = max(x0 - mg, 0), min(x0 + tp + mg, W)
        by, bx = y0 - ya, x0 - xa                        # boundary inside the crop
        base = f"tile_{k:03d}_{c['bin'].replace(' ', '')}_pred{c['pred']}"
        img = stretch(gray[ya:yb, xa:xb])

        def boundary(rgb):
            """Draw the real tile edge. Count seeds whose centre is inside it."""
            g = (90, 230, 120)
            for t in range(3):
                for yy in (by + t, min(by + tp - t, rgb.shape[0] - 1)):
                    if 0 <= yy < rgb.shape[0]:
                        rgb[yy, max(bx, 0):min(bx + tp, rgb.shape[1])] = g
                for xx in (bx + t, min(bx + tp - t, rgb.shape[1] - 1)):
                    if 0 <= xx < rgb.shape[1]:
                        rgb[max(by, 0):min(by + tp, rgb.shape[0]), xx] = g
            return rgb

        Image.fromarray(boundary(np.dstack([img] * 3))).save(os.path.join(out, base + ".png"))

        rgb = np.dstack([img] * 3)                       # mark what SeedSizer found
        if len(keep):
            m = ((cy >= y0) & (cy < y0 + tp) & (cx >= x0) & (cx < x0 + tp))
            for _, o in keep[m].iterrows():
                ry, rx = int(o["centroid-0"] - ya), int(o["centroid-1"] - xa)
                col = (255, 60, 60) if o["is_clump"] else (60, 200, 255)
                r = 11
                yy0, yy1 = max(ry - r, 0), min(ry + r, rgb.shape[0] - 1)
                xx0, xx1 = max(rx - r, 0), min(rx + r, rgb.shape[1] - 1)
                rgb[yy0:yy1, xx0] = col; rgb[yy0:yy1, xx1] = col
                rgb[yy0, xx0:xx1] = col; rgb[yy1, xx0:xx1] = col
        Image.fromarray(boundary(rgb)).save(os.path.join(out, base + "_marked.png"))

        rows.append({"tile": base, "bin": c["bin"], "row": c["iy"], "col": c["ix"],
                     "seedsizer_count": c["pred"], "objects": c["objects"],
                     "clumps": c["clumps"], "my_count": ""})

    with open(os.path.join(out, "counts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_html(out, os.path.basename(scan), rows, tile_mm, total, degenerate)
    print(f"\nopen: {os.path.join(out, 'index.html')}")


def write_html(out, scan, rows, tile_mm, total, degenerate):
    cards = []
    for r in rows:
        cards.append(f"""
    <div class="card">
      <div class="hd"><b>{html.escape(r['tile'])}</b>
        <span class="bin">{html.escape(r['bin'])}</span></div>
      <a href="{r['tile']}.png" target="_blank"><img src="{r['tile']}.png" loading="lazy"></a>
      <div class="row">
        <span>SeedSizer: <b>{r['seedsizer_count']}</b></span>
        <span>objects {r['objects']} &middot; clumps {r['clumps']}</span>
      </div>
      <div class="row"><a href="{r['tile']}_marked.png" target="_blank">what it detected &rarr;</a></div>
    </div>""")
    doc = f"""<title>Tile check - {html.escape(scan)}</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:24px;background:#14110f;color:#eee}}
 h1{{font-size:18px;margin:0 0 4px}} .sub{{color:#aa9;margin-bottom:18px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}}
 .card{{background:#221d19;border:1px solid #3a322c;border-radius:8px;padding:10px}}
 .card img{{width:100%;display:block;border-radius:4px;background:#000}}
 .hd{{display:flex;justify-content:space-between;margin-bottom:6px;font-size:12px}}
 .bin{{color:#e0a060}} .row{{display:flex;justify-content:space-between;font-size:12px;margin-top:6px;color:#cbb}}
 a{{color:#7cc7ff}} code{{background:#2c2520;padding:2px 5px;border-radius:3px}}
</style>
<h1>Tile check &mdash; {html.escape(scan)}</h1>
<div class="sub">{len(rows)} tiles of {tile_mm:.0f}&times;{tile_mm:.0f} mm &middot;
 whole-scan total <b>{total}</b> seeds
 {'&middot; <b style="color:#e08060">Otsu was degenerate; fixed threshold used</b>' if degenerate else ''}
 <br>Click a tile to enlarge and count it. Put your number in the
 <code>my_count</code> column of <code>counts.csv</code>, then run
 <code>python tiles.py score TileCheck/&lt;folder&gt;</code>.
 <br>On the detected view: blue box = counted as one seed, red box = counted as a clump
 (worth more than one).
 <br><b>The green line is the tile boundary.</b> Whole seeds are shown past it for
 context - count only the ones whose CENTRE falls inside the line, which is the
 same rule SeedSizer is scored by, so an edge seed is never ambiguous.</div>
<div class="grid">{''.join(cards)}
</div>"""
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(doc)


def score(tiles_dir):
    path = os.path.join(tiles_dir, "counts.csv")
    if not os.path.exists(path):
        sys.exit(f"no {path}")
    rows = [r for r in csv.DictReader(open(path)) if r["my_count"].strip() != ""]
    if not rows:
        sys.exit("no counts entered yet: fill in the my_count column of counts.csv")
    pred = np.array([float(r["seedsizer_count"]) for r in rows])
    mine = np.array([float(r["my_count"]) for r in rows])
    print(f"{len(rows)} tiles counted   your total {int(mine.sum())}   "
          f"SeedSizer total {int(pred.sum())}   ratio {pred.sum()/max(mine.sum(),1):.3f}\n")
    print(f"{'tile':34s} {'yours':>6s} {'seedsizer':>10s} {'diff':>6s}")
    for r, m, p in zip(rows, mine, pred):
        flag = "  <-- off by >20%" if m and abs(p / m - 1) > 0.2 else ""
        print(f"{r['tile']:34s} {int(m):6d} {int(p):10d} {int(p-m):+6d}{flag}")
    nz = mine > 0
    if nz.any():
        r = pred[nz] / mine[nz]
        print(f"\nper-tile ratio: median {np.median(r):.3f}  "
              f"within10% {100*np.mean(np.abs(r-1)<=.1):.0f}%  "
              f"worst {r.max():.2f}x / {r.min():.2f}x")


if __name__ == "__main__":
    cmds = {"sample": sample, "score": score}
    if len(sys.argv) < 3 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]](*sys.argv[2:])
