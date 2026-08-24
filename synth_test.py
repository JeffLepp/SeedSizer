"""Synthetic seed scans with known ground truth, and a screen that runs the real
SeedSizer pipeline over them.

    python synth_test.py make [reps] [seed]     write SynthScans/*.png + truth.csv
    python synth_test.py screen [tol] [only]    score threshold candidates
    python synth_test.py batch [n] [bed] [keep]  n randomised scans, truth vs measured
    python synth_test.py verify [real_dir]      compare synthetic stats to real scans
    python synth_test.py gui                    interactive synthetic scan producer
    python synth_test.py demo                   self-check (fast)

WHY THIS EXISTS
Counts from a scan cannot be checked against a scan -- you would be grading the
answer with the answer. Here the seeds are drawn, so the count and every seed's
area are known exactly, and a method is right or wrong with nothing to argue
about. Screening drives the real SeedSizer.Run(), filters and clump division
included, so the harness cannot drift from what ships.

WHAT IT IS CALIBRATED AGAINST
Everything below was measured on real 1200 PPI scans, not guessed, and `verify`
re-checks it on demand. Getting this wrong is not cosmetic: an earlier version
drew seeds too bright and on too small a canvas, and it recommended a threshold
that inflated a healthy scan by 36%. Recalibrating changed the answer. If you
move this to a different scanner, run `verify` and expect to retune.

COST
Each scan is a real A4 bed at 1200 PPI: 10200x14040 = 143 Mpx. Generation needs
roughly 1.2 GB of RAM and produces ~80 MB per PNG (about 700 MB for a full set).
`screen` runs one full SeedSizer pass per method per scan -- budget about a
minute each, so a full sweep is hours, not minutes. `demo` uses a small bed and
takes seconds.
"""
import io, os, sys, csv, glob, contextlib, threading, time, traceback
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None     # a 143 Mpx bed trips Pillow's bomb guard

PPI = 1200
PP_SQMM = (PPI / 25.4) ** 2
PX_PER_MM = PPI / 25.4
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "SynthScans")

# ---------------------------------------------------------------------------
# Calibration. Measured on the six real scans; `verify` re-checks these.
#   background median 29.3-30.3, background MAD 2.97-3.95, row-mean sd 1.6-2.1
#   brightest seed pixel 124-135 on the healthy pots, but only 70.7 on Pot_208
# Real camelina on this scanner is DARK. Drawing it beige makes every method
# look good and hides the case that actually breaks them.
SEED_AREA_MM2 = 1.45          # healthy camelina; matches the four good scans
SEED_AREA_CV = 0.28
SEED_ASPECT = (1.25, 1.9)     # major/minor
HEALTHY_GRAY, STRESSED_GRAY = (86, 132), (48, 72)

BG_LEVEL = 30.6               # pre-vignette; lands at ~29.8 measured, matching real 29.3-30.3
# Real scans are RGB uint8 with a warm cast (channel medians ~41/34.5/31), and
# SeedSizer averages the channels. That average lands on thirds, so a real MAD
# can be 3.46; a single-channel uint8 image can only produce 2.97 or 4.45. Since
# the cut is background + k*MAD, the same k lands 3 gray levels apart on the two
# -- which is the whole difference between k=3 and k=4. The synthetic scans must
# therefore be RGB, with the sensor noise added per channel, as it really is.
# Measured on the real scans: seed pixels average R:G:B = 1.00:0.70:0.56 and
# background 1.00:0.61:0.52. Both are warm, so the cast is a channel GAIN, not an
# offset -- an additive cast cannot make seeds and background warm at once, which
# is why synthetic seeds came out gray next to the real orange ones.
CH_GAIN = (1.37, 0.89, 0.74)  # normalised so the three average to 1.0
CH_NOISE = 3.00               # per-channel white sd
CH_CORR = 0.70                # measured correlation between channels
# Real MAD is not one number: across the six scans it takes 3.459 and 3.954 (2
# and 2/3 vs 2 and 1/3, on thirds). The cut being background + k*MAD, that 14%
# spread moves a k=4 threshold from 44.1 to 46.1. A generator pinned to one MAD
# would be testing a single lucky scanner day, so jitter it per scan.
CH_NOISE_JITTER = (0.90, 1.14)
# Real background noise is NOT pixel-independent. i.i.d. noise leaves single
# stray pixels, which remove_small_objects() deletes for free; correlated noise
# leaves blobs, which survive and get counted. This is the difference between a
# threshold looking safe here and inflating a real scan.
BG_CORR_SD = 1.20             # low-frequency component, shared by all channels
BG_CORR_PX = 24               # correlation length in pixels

# Debris. Pot_208 has 0.048% of its pixels above 5 MAD in a FLAT tail -- about
# 54,000 pixels of material that is not seed and not Gaussian noise. The size
# distribution has to straddle the 0.1 mm^2 filter and the seed-size floor,
# because the objects that cross those floors as the threshold drops are exactly
# what turns a good threshold into an overcount.
DUST_AREA_MM2 = 0.05          # median; the CV supplies the long tail
DUST_AREA_CV = 1.40
# Brightness is skewed, not uniform: most debris is barely above background and a
# few specks are far brighter than any seed. The rare bright end is not
# decoration -- it is what drags yen to 81 on Pot_208, above every real seed
# there, which is how yen ends up counting nothing on that scan.
DUST_GRAY_BASE = 34.0
DUST_GRAY_MED = 9.0           # median rise above base
DUST_GRAY_CV = 0.85
DUST_GRAY_MAX = 150.0
DUST_FEATHER = 3.5            # soft edges swell as the cut drops

# The real beds are far cleaner than scattered dust would suggest, but Pot_200
# and Pot_208 both carry a bright striped rectangle near centre-bottom - a
# scanner artifact from the same flat. That one object, not a sprinkling of
# debris, is where much of the bright-pixel tail comes from, and being large it
# survives every size filter. Modelling it as dust was the right statistic
# reached by the wrong mechanism.
ARTIFACT_MM = (13.0, 17.0)    # width, height
ARTIFACT_GRAY = (52.0, 74.0)  # dim, but well above background
ARTIFACT_STRIPE_PX = 60       # vertical striping, wide enough to be visible as in the scans
# It shows on some flats and not others (Pot_200 and Pot_208 have it, Pot_95 does
# not), so roughly half of full beds. A smaller test bed is in effect a CROP of a
# real bed, and a crop only contains the artifact as often as its area implies -
# without that scaling, one fixed-size rectangle swamps a small bed and its clump
# division alone invents ~150 phantom seeds.
ARTIFACT_PROB = 0.5

# Illumination is three broad horizontal zones with fairly sharp seams, not a
# smooth sinusoid, plus a gentle darkening toward the edges of the bed.
BAND_ZONES = 3
BAND_SEAM_PX = 90             # how abruptly one zone gives way to the next
BAND_SHAPE = (-0.55, 0.85, -0.30)   # relative zone levels; the pattern repeats between scans
VIGNETTE = 0.055              # fractional darkening at the extreme corners

# Every real scan is the same A4 bed. Not a detail: Pot_208 is degenerate
# BECAUSE 12 seeds share a histogram with 143 million background pixels, and no
# smaller canvas reproduces that ratio at any seed count.
BED = (14040, 10200)
SEED_REGION = (0.04, 0.72, 0.03, 0.97)   # top, bottom, left, right as bed fractions

# name: (n_seeds, frac_in_clusters, cluster_size, band_amp, n_dust, seed_scale, gray)
# Counts track the real sscounts (524-1300 on the healthy pots, 12 on Pot_208).
# band_amp 2.5 reproduces the real row-mean sd of ~1.8; 6.0 is a deliberate
# 3x-worst-observed stress case, not something any scan on hand actually does.
PRESETS = {
    "healthy":     (900,  0.00, (0, 0),   2.5,  240, 1.00, HEALTHY_GRAY),
    "touching":    (900,  0.60, (2, 8),   2.5,  240, 1.00, HEALTHY_GRAY),
    "heavy_clump": (900,  0.95, (8, 40),  2.5,  240, 1.00, HEALTHY_GRAY),
    "sparse":      (12,   0.00, (0, 0),   2.5,  120, 0.60, STRESSED_GRAY),  # like Pot_208
    "sparse_clump":(40,   0.70, (2, 6),   2.5,  120, 0.60, STRESSED_GRAY),
    "field":       (1150, 0.40, (2, 8),   2.5, 1400, 1.00, HEALTHY_GRAY),   # WOCS: chaffy
    "banded":      (600,  0.30, (2, 5),   6.0,  240, 1.00, HEALTHY_GRAY),
    "shrivelled":  (900,  0.40, (2, 8),   2.5,  240, 0.55, STRESSED_GRAY),  # heat-stressed
    "dense_dark":  (1200, 0.50, (3, 12),  2.5,  240, 1.00, (66, 104)),
}

# Occupancy is tracked at the half-alpha contour, where the geometric ellipse
# boundary sits. Measuring the union anywhere else makes it disagree with the sum
# of pi*a*b areas and invents overlap that is not there.
CONTOUR = 0.5
DEGENERATE_FG = 0.15          # >15% of a scan being seed is physically impossible


# ------------------------------------------------------------------ drawing

def ellipse(a, b, theta, feather=1.6):
    """Antialiased ellipse alpha mask, 1.0 inside, feathered over ~feather px."""
    r = int(np.ceil(max(a, b))) + 3
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    c, s = np.cos(theta), np.sin(theta)
    u, v = (x * c + y * s) / a, (-x * s + y * c) / b
    d = np.sqrt(u * u + v * v)                       # 1.0 on the rim
    return np.clip((1.0 - d) * (min(a, b) / feather) + 0.5, 0.0, 1.0)


def _axes(rng, area_mm2, cv, aspect):
    area_px = max(rng.lognormal(np.log(area_mm2), cv), 0.004) * PP_SQMM
    ar = rng.uniform(*aspect)
    b = np.sqrt(area_px / (np.pi * ar))
    return b * ar, b, rng.uniform(0, np.pi)


def _box(shape, mask, cy, cx):
    r = mask.shape[0] // 2
    y0, x0 = cy - r, cx - r
    if y0 < 0 or x0 < 0 or y0 + mask.shape[0] > shape[0] or x0 + mask.shape[1] > shape[1]:
        return None
    return (slice(y0, y0 + mask.shape[0]), slice(x0, x0 + mask.shape[1]))


def _blit(img, occ, mask, cy, cx, gray, mark=True):
    """Alpha-composite one blob. Returns False if it would leave the bed."""
    sl = _box(img.shape, mask, cy, cx)
    if sl is None:
        return False
    img[sl] = img[sl] * (1 - mask) + gray * mask
    if mark:
        occ[sl] |= mask > CONTOUR
    return True


def _free(occ, mask, cy, cx):
    sl = _box(occ.shape, mask, cy, cx)
    return False if sl is None else not occ[sl][mask > CONTOUR].any()


def _background(img, rng, band, H, W):
    """Banding plus low-frequency noise. White sensor noise is NOT added here --
    it is per channel and goes on at RGB expansion, which is where it physically
    happens. Written in row blocks so no temporary approaches the bed's size."""
    from scipy.ndimage import zoom
    rows = np.arange(H, dtype=np.float64)
    # Three zones with tanh seams, then a small ripple on top.
    edges = np.linspace(0, H, BAND_ZONES + 1)[1:-1]
    # The illumination pattern is a property of the scanner, so its shape repeats;
    # only its depth jitters. Centring the levels on zero keeps the background
    # MEDIAN stable, which is what the real scans show (29.3-30.7 across six).
    levels = np.array(BAND_SHAPE) + rng.uniform(-0.25, 0.25, BAND_ZONES)
    levels -= levels.mean()
    prof = np.full(H, levels[0])
    for e, lv in zip(edges, levels[1:]):
        prof += (lv - prof[int(min(e, H - 1))]) * 0.5 * (1 + np.tanh((rows - e) / BAND_SEAM_PX))
    prof = (BG_LEVEL + band * prof
            + 0.25 * band * np.sin(2 * np.pi * rows / rng.uniform(600, 1600) + rng.uniform(0, 6.3)))
    lo = rng.normal(0, BG_CORR_SD, (H // BG_CORR_PX + 3, W // BG_CORR_PX + 3)).astype(np.float32)
    for r0 in range(0, H, 1024):
        r1 = min(r0 + 1024, H)
        l0, l1 = r0 // BG_CORR_PX, r1 // BG_CORR_PX + 2
        slab = zoom(lo[l0:l1], BG_CORR_PX, order=1)
        off = r0 - l0 * BG_CORR_PX
        corr = slab[off:off + (r1 - r0), :W]
        if corr.shape != (r1 - r0, W):               # never expected; do not fake data
            raise RuntimeError(f"correlated-noise slab {corr.shape} != {(r1 - r0, W)}")
        img[r0:r1] = prof[r0:r1, None] + corr


def _to_rgb(scene, rng):
    """Luminance scene -> RGB uint8, with the cast and the partly-shared
    per-channel sensor noise measured on the real scans."""
    H, W = scene.shape
    rgb = np.empty((H, W, 3), np.uint8)
    sd = CH_NOISE * rng.uniform(*CH_NOISE_JITTER)
    a = np.sqrt(CH_CORR) * sd                # shared part
    b = np.sqrt(1.0 - CH_CORR) * sd          # independent part
    for r0 in range(0, H, 1024):
        r1 = min(r0 + 1024, H)
        block = scene[r0:r1]
        shared = rng.normal(0, a, block.shape)
        for c in range(3):
            v = block * CH_GAIN[c] + shared + rng.normal(0, b, block.shape)
            rgb[r0:r1, :, c] = np.clip(v, 0, 255).astype(np.uint8)
    return rgb


def generate(preset, rng, bed=BED, params=None):
    """Draw one scan. Returns (uint8 RGB image, truth dict).

    params overrides the preset table with an explicit 7-tuple, which is how
    `batch` draws randomised scans without inventing preset names."""
    if params is None:
        if preset not in PRESETS:
            raise KeyError(f"unknown preset {preset!r}; have {sorted(PRESETS)}")
        params = PRESETS[preset]
    n, frac, csize, band, n_dust, scale, gray = params
    H, W = bed

    img = np.empty((H, W), np.float32)
    _background(img, rng, band, H, W)
    occ = np.zeros((H, W), bool)
    cocc = np.zeros((H, W), bool)                    # cluster-local: members can't stack

    # On the real flats the seed is poured into a tray, so it occupies a central
    # band rather than the whole bed. This changes local density, which is what
    # the reference-seed-area median actually sees.
    ry0, ry1 = int(SEED_REGION[0] * H), int(SEED_REGION[1] * H)
    rx0, rx1 = int(SEED_REGION[2] * W), int(SEED_REGION[3] * W)

    n_cluster = int(round(n * frac))
    areas, placed, touching = [], 0, 0
    seed_kw = dict(area_mm2=SEED_AREA_MM2 * scale, cv=SEED_AREA_CV, aspect=SEED_ASPECT)

    attempts = 0
    while placed < n_cluster and attempts < 40 * n:
        attempts += 1
        k = min(int(rng.integers(csize[0], csize[1] + 1)), n_cluster - placed)
        pend, seeds = [], []
        for j in range(k):
            a, b, th = _axes(rng, **seed_kw)
            m, grow = ellipse(a, b, th), ellipse(a + 4, b + 4, th)
            if j == 0:
                cy = None
                for _ in range(40):                  # a bed-edge miss must not
                    ty, tx = int(rng.integers(ry0, ry1)), int(rng.integers(rx0, rx1))
                    if _free(occ, grow, ty, tx):     # abandon the whole cluster
                        cy, cx = ty, tx
                        break
                if cy is None:
                    break
            else:
                # Slide outward from a seated member until the footprints separate,
                # then back off 1.5 px so they abut. Real seeds touch; they do not
                # share pixels, so a clump's area is the sum of its seeds'.
                cy = None
                for _ in range(8):                   # retry other parents/angles
                    py, px, pa, pb = seeds[rng.integers(len(seeds))]
                    sn, cs = np.sin(a1 := rng.uniform(0, 2 * np.pi)), np.cos(a1)
                    d = 0.45 * (max(a, b) + max(pa, pb))
                    for _ in range(80):
                        ty, tx = int(py + d * sn), int(px + d * cs)
                        if _free(cocc, m, ty, tx) and _free(occ, grow, ty, tx):
                            break
                        d += 2.0
                    else:
                        continue
                    ty, tx = int(py + (d - 1.5) * sn), int(px + (d - 1.5) * cs)
                    if _free(occ, grow, ty, tx):
                        cy, cx = ty, tx
                        break
                if cy is None:
                    break
            cocc[_box(cocc.shape, m, cy, cx)] |= m > CONTOUR
            pend.append((m, cy, cx, rng.uniform(*gray)))
            seeds.append((cy, cx, a, b))
        for m, cy, cx, g in pend:
            _blit(img, occ, m, cy, cx, g)
            areas.append(float(m.sum()) / PP_SQMM)
            cocc[_box(cocc.shape, m, cy, cx)] = False
        placed += len(pend)
        touching += len(pend) if len(pend) > 1 else 0

    tries = 0
    while placed < n and tries < 60 * n:             # loners: never allowed to touch
        tries += 1
        a, b, th = _axes(rng, **seed_kw)
        m, grow = ellipse(a, b, th), ellipse(a + 4, b + 4, th)
        cy, cx = int(rng.integers(ry0, ry1)), int(rng.integers(rx0, rx1))
        if not _free(occ, grow, cy, cx):
            continue
        _blit(img, occ, m, cy, cx, rng.uniform(*gray))
        areas.append(float(m.sum()) / PP_SQMM)
        placed += 1

    if placed < n:
        print(f"    note: {preset} seated {placed}/{n} seeds; truth records {placed}", flush=True)

    for _ in range(n_dust):                          # debris is NOT in the truth count
        a, b, th = _axes(rng, DUST_AREA_MM2, DUST_AREA_CV, (1.0, 2.6))
        m = ellipse(max(a, 1.5), max(b, 1.2), th, DUST_FEATHER)
        cy, cx = int(rng.integers(H)), int(rng.integers(W))
        g = DUST_GRAY_BASE + rng.lognormal(np.log(DUST_GRAY_MED), DUST_GRAY_CV)
        _blit(img, occ, m, cy, cx, min(g, DUST_GRAY_MAX), mark=False)

    # One scanner artifact, as seen on the real flats: a dim striped rectangle.
    aw = int(ARTIFACT_MM[0] * PX_PER_MM)
    ah = int(ARTIFACT_MM[1] * PX_PER_MM)
    bed_frac = min(1.0, (H * W) / float(BED[0] * BED[1]))
    if aw < W and ah < H and rng.random() < ARTIFACT_PROB * bed_frac:
        ay = int(rng.uniform(0.45, 0.72) * (H - ah))
        ax = int(rng.uniform(0.30, 0.60) * (W - aw))
        lo_g, hi_g = ARTIFACT_GRAY
        stripe = lo_g + (hi_g - lo_g) * (
            0.5 + 0.5 * np.sin(2 * np.pi * np.arange(aw) / ARTIFACT_STRIPE_PX))
        patch = np.repeat(stripe[None, :], ah, axis=0)
        edge = np.minimum(np.minimum(np.arange(ah), ah - 1 - np.arange(ah))[:, None],
                          np.minimum(np.arange(aw), aw - 1 - np.arange(aw))[None, :])
        alpha = np.clip(edge / 30.0, 0, 1) * 0.75      # it is a soft patch, not a decal
        sub = img[ay:ay + ah, ax:ax + aw]
        img[ay:ay + ah, ax:ax + aw] = sub * (1 - alpha) + patch * alpha

    # Gentle corner falloff, as on the real beds.
    yy = (np.arange(H) / (H - 1) - 0.5) * 2
    xx = (np.arange(W) / (W - 1) - 0.5) * 2
    for r0 in range(0, H, 1024):
        r1 = min(r0 + 1024, H)
        v = 1.0 - VIGNETTE * (yy[r0:r1, None] ** 2 + xx[None, :] ** 2) / 2.0
        img[r0:r1] *= v

    # occ holds the seed union only. Against the sum of the seeds' own areas it
    # says how much footprint the clusters swallowed -- which must be ~0.
    union = float(occ.sum()) / PP_SQMM
    total = float(np.sum(areas))
    truth = {"preset": preset, "bed": f"{W}x{H}", "n_seeds": placed, "n_touching": touching,
             "n_dust": n_dust, "mean_area_mm2": round(float(np.mean(areas)), 4),
             "total_area_mm2": round(total, 3),
             "overlap_pct": round(100 * (1 - union / total), 2),
             "fill_pct": round(100 * float(occ.mean()), 3)}
    return _to_rgb(img, rng), truth


TRUTH_COLS = ["file", "preset", "bed", "n_seeds", "n_touching", "n_dust",
              "mean_area_mm2", "total_area_mm2", "overlap_pct", "fill_pct"]


def make(reps=1, seed=7):
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for preset in PRESETS:
        for r in range(reps):
            img, truth = generate(preset, rng)
            truth["file"] = f"{preset}_{r}.png"
            Image.fromarray(img).save(os.path.join(OUT, truth["file"]))
            rows.append(truth)
            print(f"  {truth['file']:20s} seeds={truth['n_seeds']:5d} "
                  f"touching={truth['n_touching']:5d}  fill={truth['fill_pct']:.2f}%  "
                  f"overlap={truth['overlap_pct']:+.1f}%", flush=True)
            del img
    path = os.path.join(OUT, "truth.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, TRUTH_COLS)
        w.writeheader()
        w.writerows([{c: t[c] for c in TRUTH_COLS} for t in rows])
    print(f"\nwrote {path}  ({len(rows)} scans)")


# ---------------------------------------------------------------- screening

def _mad(s):
    med = float(np.median(s))
    return med, float(np.median(np.abs(s - med))) * 1.4826


def _candidates():
    from skimage.filters import threshold_otsu, threshold_triangle, threshold_yen

    def guarded(sub, fallback):
        t = threshold_otsu(sub)
        return t if float((sub > t).mean()) <= DEGENERATE_FG else fallback(sub)

    def mad(k):
        return lambda sub: (lambda m: m[0] + k * m[1])(_mad(sub))

    fixed = lambda sub: 62.0
    # Stressed seed sits at gray 45-75 over a 29.7 +/- 3.46 background, so the cut
    # has to land near 40 to see it at all. Anything at 6 MAD or above (50.5+) is
    # already past the dimmest seeds on the scan.
    c = {"otsu": threshold_otsu, "triangle": threshold_triangle, "yen": threshold_yen,
         "fixed_62": fixed, "guarded->fixed_62": lambda s: guarded(s, fixed)}
    for k in (3, 4, 5, 6, 8):
        c[f"bg+{k}mad"] = mad(k)
    for k in (3, 4):
        c[f"guarded->bg+{k}mad"] = (lambda kk: lambda s: guarded(s, mad(kk)))(k)
    return c


def _load_truth():
    path = os.path.join(OUT, "truth.csv")
    if not os.path.exists(path):
        sys.exit(f"no {path}\nrun `python synth_test.py make` first.")
    rows = list(csv.DictReader(open(path)))
    missing = [r["file"] for r in rows if not os.path.exists(os.path.join(OUT, r["file"]))]
    if missing:
        sys.exit(f"truth.csv lists {len(missing)} scans that are not on disk "
                 f"(e.g. {missing[0]}).\nre-run `python synth_test.py make`.")
    return rows


def screen(tol=0.10, only=""):
    sys.path.insert(0, HERE)          # import the SeedSizer next to this file
    import SeedSizer as S
    if not hasattr(S, "Run"):
        sys.exit(f"imported {S.__file__!r}, which has no Run(). Wrong SeedSizer on sys.path.")

    truth = _load_truth()
    keep = set(only.split(",")) if only else None
    cands = {k: v for k, v in _candidates().items() if keep is None or k in keep}
    if keep and not cands:
        sys.exit(f"no candidate matched {only!r}; have {sorted(_candidates())}")

    original = S.threshold_otsu
    rows = []
    try:
        for name, fn in cands.items():
            S.threshold_otsu = lambda img, _f=fn: float(_f(img[::4, ::4]))
            for t in truth:
                with contextlib.redirect_stdout(io.StringIO()):
                    r = S.Run(os.path.join(OUT, t["file"]))
                true_n, true_a = int(t["n_seeds"]), float(t["mean_area_mm2"])
                got_n = int(r["sscount"])
                got_a = float(r["AvgSizeOfOneSeed"]) if r["AvgSizeOfOneSeed"] != "" else float("nan")
                rows.append({"method": name, "file": t["file"], "preset": t["preset"],
                             "true_count": true_n, "count": got_n,
                             "count_ratio": round(got_n / true_n, 4) if true_n else float("nan"),
                             "true_area": true_a, "area": round(got_a, 4),
                             "area_ratio": round(got_a / true_a, 4) if true_a else float("nan")})
                print(f"  {name:18s} {t['file']:20s} {true_n:5d} -> {got_n:7d} "
                      f"({rows[-1]['count_ratio']:.2f}x)", flush=True)
    finally:
        S.threshold_otsu = original   # leave the module as we found it

    out = os.path.join(OUT, "SynthScreen.csv")
    if keep is not None and os.path.exists(out):      # merge, don't clobber the rest
        prior = [r for r in csv.DictReader(open(out)) if r["method"] not in keep]
        for r in prior:
            for k in ("true_count", "count"):
                r[k] = int(r[k])
            for k in ("count_ratio", "true_area", "area", "area_ratio"):
                r[k] = float(r[k])
        rows = prior + rows
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}\n")
    report(rows, tol)


def report(rows, tol=0.10):
    lo, hi = 1 - tol, 1 + tol
    by = {}
    for r in rows:
        m = by.setdefault(r["method"], {"pass": 0, "n": 0, "worst": 1.0, "where": ""})
        m["n"] += 1
        ratio = r["count_ratio"]
        m["pass"] += lo <= ratio <= hi
        err = ratio if ratio >= 1 else (1 / ratio if ratio > 0 else np.inf)
        if err > m["worst"]:
            m["worst"], m["where"] = err, r["preset"]
    print(f"===== count within {tol:.0%} of truth =====")
    print(f"{'method':20s} {'pass':>9s} {'worst':>10s}  worst case")
    for name, m in sorted(by.items(), key=lambda kv: (-kv[1]["pass"], kv[1]["worst"])):
        print(f"{name:20s} {m['pass']:4d}/{m['n']:<4d} {m['worst']:9.2f}x  {m['where']}")


# ------------------------------------------------------------------ verify

def _stats(path_or_arr, label):
    from skimage.filters import threshold_otsu, threshold_yen
    a = path_or_arr
    if isinstance(a, str):
        a = np.asarray(Image.open(a))
    if a.ndim == 3:
        a = a[:, :, :3].mean(2)
    a = a[::4, ::4].astype(np.float32)
    med, mad = _mad(a)
    o = float(threshold_otsu(a))
    with np.errstate(divide="ignore", invalid="ignore"):
        yen = float(threshold_yen(a))
    return {"scan": label, "med": med, "MAD": mad, "otsu": o, "yen": yen,
            "fg@otsu%": 100 * float((a > o).mean()), "p99.9": float(np.percentile(a, 99.9)),
            "max": float(a.max()),
            ">3MAD%": 100 * float((a > med + 3 * mad).mean()),
            ">4MAD%": 100 * float((a > med + 4 * mad).mean()),
            ">5MAD%": 100 * float((a > med + 5 * mad).mean())}


def verify(real_dir=""):
    """Print synthetic and real scan statistics side by side.

    The synthetic set only means something while these agree. Re-run this after
    any scanner change; if the columns diverge, the calibration constants at the
    top of this file are stale and the screen's ranking cannot be trusted.
    """
    rows = [_stats(p, "SYN " + os.path.basename(p)[:-4])
            for p in sorted(glob.glob(os.path.join(OUT, "*.png")))]
    if real_dir:
        real = sorted(glob.glob(os.path.join(real_dir, "*.tif"))
                      + glob.glob(os.path.join(real_dir, "*.tiff")))
        if not real:
            sys.exit(f"no .tif/.tiff under {real_dir!r}")
        import imageio.v3 as iio
        for p in real:
            rows.append(_stats(iio.imread(p), "REAL " + os.path.basename(p)[:28]))
    if not rows:
        sys.exit(f"nothing to verify: no PNGs in {OUT} and no real_dir given.")

    cols = [c for c in rows[0] if c != "scan"]
    print(f"{'scan':34s} " + " ".join(f"{c:>9s}" for c in cols))
    for r in rows:
        print(f"{r['scan']:34s} " + " ".join(f"{r[c]:9.3f}" for c in cols))
    print("\nSynthetic and real should agree on med, MAD and the >kMAD tail.\n"
          "The tail is what decides k in bg+k*MAD, so a mismatch there is not cosmetic.")


# ------------------------------------------------------------------- batch

def _random_params(rng, bed_px):
    """One randomised but physically plausible scan.

    Ranges are deliberately wider than the nine fixed presets: the point of a
    batch is to find where accuracy falls off, not to re-pass known cases."""
    bed_mm2 = (bed_px / PX_PER_MM) ** 2
    scale = float(rng.uniform(0.45, 1.20))            # shrivelled .. plump
    fill = float(rng.uniform(0.0015, 0.040))          # near-empty .. densely covered
    n = max(3, int(bed_mm2 * fill / (SEED_AREA_MM2 * scale)))
    lo = float(rng.uniform(40, 78))                   # dim stressed .. bright plump
    gray = (lo, min(lo + float(rng.uniform(20, 46)), 150.0))
    frac = float(rng.choice([0.0, 0.15, 0.35, 0.60, 0.85, 0.95]))
    hi = int(rng.integers(3, 26))
    n_dust = int(bed_mm2 * rng.uniform(0.002, 0.035))
    return (n, frac, (2, hi), float(rng.uniform(1.5, 6.0)), n_dust, scale, gray)


def batch(n=100, bed=3000, keep=6, seed=11):
    """Generate n randomised scans, run the real SeedSizer on each, and record
    truth against measurement. PNGs past `keep` are deleted as we go so a long
    run does not fill the disk."""
    sys.path.insert(0, HERE)
    import SeedSizer as S
    if not hasattr(S, "Run"):
        sys.exit(f"imported {S.__file__!r}, which has no Run().")
    os.makedirs(OUT, exist_ok=True)

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(int(n)):
        params = _random_params(rng, bed)
        img, t = generate("_random", rng, (int(bed), int(bed)), params)
        name = f"batch_{i:03d}.png"
        path = os.path.join(OUT, name)
        Image.fromarray(img).save(path)
        del img
        with contextlib.redirect_stdout(io.StringIO()):
            r = S.Run(path)
        got, true_n = int(r["sscount"]), t["n_seeds"]
        area = r["AvgSizeOfOneSeed"]
        rows.append({"file": name, "true_count": true_n, "count": got,
                     "count_ratio": round(got / true_n, 4) if true_n else "",
                     "true_area": t["mean_area_mm2"],
                     "area": round(float(area), 4) if area != "" else "",
                     "area_ratio": round(float(area) / t["mean_area_mm2"], 4) if area != "" else "",
                     "n_touching": t["n_touching"], "fill_pct": t["fill_pct"],
                     "n_dust": t["n_dust"], "note": r["ProcessingNote"]})
        print(f"  [{i+1}/{n}] {name}  truth={true_n:5d} got={got:6d} "
              f"({rows[-1]['count_ratio'] or 'n/a'}x)  fill={t['fill_pct']:.2f}%", flush=True)
        if i >= int(keep):
            os.remove(path)

    out = os.path.join(OUT, "BatchResults.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    batch_report(rows)


def batch_report(rows):
    rat = np.array([r["count_ratio"] for r in rows if r["count_ratio"] != ""], float)
    print(f"\n===== {len(rat)} randomised scans =====")
    for tol in (0.02, 0.05, 0.10, 0.20):
        print(f"  within {tol:4.0%}: {100 * np.mean(np.abs(rat - 1) <= tol):5.1f}%")
    print(f"  median ratio {np.median(rat):.3f}   worst over {rat.max():.2f}x   "
          f"worst under {rat.min():.2f}x")


# ---------------------------------------------------------------------- GUI

def _gui_seed_gray(brightness):
    """Map a user-facing visibility slider to the generator's gray range."""
    brightness = float(np.clip(brightness, 0, 100))
    lo = 42.0 + 0.48 * brightness
    hi = min(155.0, lo + 18.0 + 0.22 * brightness)
    return lo, hi


def _gui_brightness_from_gray(gray):
    return int(np.clip(round((float(gray[0]) - 42.0) / 0.48), 0, 100))


def _gui_write_truth(row):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "gui_truth.csv")
    cols = [
        "file", "preset", "bed", "n_seeds", "requested_seeds", "n_touching",
        "touching_pct", "n_dust", "seed_brightness", "seed_scale", "band_amp",
        "max_cluster_size", "mean_area_mm2", "total_area_mm2", "overlap_pct",
        "fill_pct", "measured_count", "count_error_pct", "processing_note",
    ]
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, cols)
        if not exists:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in cols})


def gui():
    """Interactive front-end for producing and checking synthetic seed scans."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from PIL import ImageTk
    except Exception as exc:
        sys.exit(f"Could not open the synthetic data GUI: {exc}")

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        sys.exit(f"Could not open the synthetic data GUI: {exc}")
    root.title("Synthetic Seed Scan Producer")
    root.geometry("1120x760")
    root.minsize(980, 650)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    main = ttk.Frame(root, padding=12)
    main.pack(fill="both", expand=True)
    main.columnconfigure(1, weight=1)
    main.rowconfigure(0, weight=1)

    controls = ttk.Frame(main, width=340)
    controls.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
    controls.columnconfigure(1, weight=1)

    preview = ttk.Frame(main)
    preview.grid(row=0, column=1, sticky="nsew")
    preview.columnconfigure(0, weight=1)
    preview.rowconfigure(0, weight=1)

    preset_var = tk.StringVar(value="custom")
    seed_var = tk.IntVar(value=120)
    touch_var = tk.DoubleVar(value=35)
    cluster_var = tk.IntVar(value=8)
    bright_var = tk.DoubleVar(value=70)
    scale_var = tk.DoubleVar(value=1.0)
    dust_var = tk.IntVar(value=160)
    band_var = tk.DoubleVar(value=2.5)
    random_var = tk.IntVar(value=7)
    bed_mode_var = tk.StringVar(value="Preview square")
    bed_size_var = tk.IntVar(value=3200)
    last = {"img": None, "truth": None, "photo": None, "path": None}

    busy_widgets = []

    def add_labeled(row, text, widget):
        ttk.Label(controls, text=text).grid(row=row, column=0, sticky="w", pady=4)
        widget.grid(row=row, column=1, sticky="ew", pady=4)
        busy_widgets.append(widget)

    row = 0
    add_labeled(row, "Preset", ttk.Combobox(
        controls, textvariable=preset_var, values=["custom"] + sorted(PRESETS),
        state="readonly"))
    preset_box = busy_widgets[-1]

    row += 1
    add_labeled(row, "Number of seeds", tk.Spinbox(
        controls, from_=1, to=5000, increment=1, textvariable=seed_var, width=10))

    row += 1
    add_labeled(row, "Touching/clumped (%)", ttk.Scale(
        controls, from_=0, to=100, variable=touch_var, orient="horizontal"))
    touch_label = ttk.Label(controls, text="")
    touch_label.grid(row=row, column=2, sticky="e", padx=(6, 0))

    row += 1
    add_labeled(row, "Max cluster size", tk.Spinbox(
        controls, from_=2, to=60, increment=1, textvariable=cluster_var, width=10))

    row += 1
    add_labeled(row, "Seed brightness", ttk.Scale(
        controls, from_=0, to=100, variable=bright_var, orient="horizontal"))
    bright_label = ttk.Label(controls, text="")
    bright_label.grid(row=row, column=2, sticky="e", padx=(6, 0))

    row += 1
    add_labeled(row, "Seed size scale", tk.Spinbox(
        controls, from_=0.35, to=1.6, increment=0.05, textvariable=scale_var, width=10))

    row += 1
    add_labeled(row, "Dust/chaff objects", tk.Spinbox(
        controls, from_=0, to=10000, increment=10, textvariable=dust_var, width=10))

    row += 1
    add_labeled(row, "Scanner banding", tk.Spinbox(
        controls, from_=0.0, to=8.0, increment=0.25, textvariable=band_var, width=10))

    row += 1
    add_labeled(row, "Random seed", tk.Spinbox(
        controls, from_=0, to=999999, increment=1, textvariable=random_var, width=10))

    row += 1
    add_labeled(row, "Bed mode", ttk.Combobox(
        controls, textvariable=bed_mode_var, values=["Preview square", "Full A4 scan"],
        state="readonly"))

    row += 1
    add_labeled(row, "Preview bed pixels", tk.Spinbox(
        controls, from_=1000, to=7000, increment=250, textvariable=bed_size_var, width=10))

    row += 1
    button_frame = ttk.Frame(controls)
    button_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(12, 6))
    button_frame.columnconfigure((0, 1), weight=1)

    generate_button = ttk.Button(button_frame, text="Generate Preview")
    save_button = ttk.Button(button_frame, text="Save PNG")
    analyze_button = ttk.Button(button_frame, text="Save + Run SeedSizer")
    choose_button = ttk.Button(button_frame, text="Save As...")
    generate_button.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=3)
    save_button.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=3)
    analyze_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=3)
    choose_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=3)
    busy_widgets.extend([generate_button, save_button, analyze_button, choose_button])

    row += 1
    status_var = tk.StringVar(value="Ready")
    ttk.Label(controls, textvariable=status_var, wraplength=330).grid(
        row=row, column=0, columnspan=3, sticky="ew", pady=(8, 4))

    row += 1
    results = tk.Text(controls, height=14, width=40, wrap="word", state="disabled")
    results.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(4, 0))
    controls.rowconfigure(row, weight=1)

    image_label = ttk.Label(preview, anchor="center")
    image_label.grid(row=0, column=0, sticky="nsew")

    def update_slider_labels(*_):
        touch_label.config(text=f"{int(touch_var.get())}%")
        lo, hi = _gui_seed_gray(bright_var.get())
        bright_label.config(text=f"{int(lo)}-{int(hi)}")

    for var in (touch_var, bright_var):
        var.trace_add("write", update_slider_labels)
    update_slider_labels()

    def set_results(text):
        results.config(state="normal")
        results.delete("1.0", tk.END)
        results.insert("1.0", text)
        results.config(state="disabled")

    def set_busy(is_busy):
        state = tk.DISABLED if is_busy else tk.NORMAL
        for widget in busy_widgets:
            try:
                widget.config(state=state)
            except tk.TclError:
                pass
        preset_box.config(state="disabled" if is_busy else "readonly")

    def selected_bed():
        if bed_mode_var.get() == "Full A4 scan":
            return BED
        px = int(bed_size_var.get())
        return px, px

    def selected_params():
        n = int(seed_var.get())
        frac = float(touch_var.get()) / 100.0
        cluster_max = max(2, int(cluster_var.get()))
        csize = (2, cluster_max) if frac > 0 else (0, 0)
        band = float(band_var.get())
        n_dust = max(0, int(dust_var.get()))
        scale = max(0.05, float(scale_var.get()))
        gray = _gui_seed_gray(bright_var.get())
        return n, frac, csize, band, n_dust, scale, gray

    def apply_preset(_event=None):
        name = preset_var.get()
        if name == "custom":
            return
        n, frac, csize, band, n_dust, scale, gray = PRESETS[name]
        seed_var.set(n)
        touch_var.set(int(round(frac * 100)))
        cluster_var.set(max(2, csize[1] if csize[1] else 8))
        band_var.set(band)
        dust_var.set(n_dust)
        scale_var.set(scale)
        bright_var.set(_gui_brightness_from_gray(gray))

    preset_box.bind("<<ComboboxSelected>>", apply_preset)

    def display_image(img):
        pil = Image.fromarray(img)
        pil.thumbnail((760, 680), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        image_label.config(image=photo)
        image_label.image = photo
        last["photo"] = photo

    def default_save_path():
        os.makedirs(OUT, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"gui_{stamp}_{int(seed_var.get())}seeds.png"
        return os.path.join(OUT, name)

    def format_results(truth, controls_row, save_path=None, measured=None, note=None):
        lines = [
            f"File: {os.path.basename(save_path) if save_path else '(not saved)'}",
            f"Canvas: {truth['bed']}",
            f"Requested seeds: {controls_row['requested_seeds']}",
            f"Placed truth seeds: {truth['n_seeds']}",
            f"Touching seeds: {truth['n_touching']} ({controls_row['touching_pct']}% requested)",
            f"Footprint overlap: {truth['overlap_pct']:+.2f}%",
            f"Fill: {truth['fill_pct']:.3f}%",
            f"Mean seed area: {truth['mean_area_mm2']:.4f} mm^2",
            f"Dust/chaff objects: {truth['n_dust']}",
            f"Seed brightness: {controls_row['seed_brightness']}",
            f"Seed size scale: {controls_row['seed_scale']}",
            f"Scanner banding: {controls_row['band_amp']}",
        ]
        if truth["n_seeds"] != controls_row["requested_seeds"]:
            lines.append("Placement note: not all requested seeds fit on this bed.")
        if measured is not None:
            got = int(measured["sscount"])
            true_n = max(1, int(truth["n_seeds"]))
            err = 100.0 * (got - true_n) / true_n
            lines.extend([
                "",
                f"SeedSizer count: {got}",
                f"Count error: {err:+.2f}%",
                f"Accepted objects: {measured['AcceptedObjectCount']}",
                f"Rejected objects: {measured['RejectedObjectCount']}",
            ])
            if measured["ProcessingNote"]:
                lines.append(f"Note: {measured['ProcessingNote']}")
        if note:
            lines.extend(["", note])
        return "\n".join(lines)

    def run_job(save=False, analyze=False, save_as=False):
        try:
            params = selected_params()
            bed = selected_bed()
            seed = int(random_var.get())
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        save_path = None
        if save or analyze or save_as:
            if save_as:
                os.makedirs(OUT, exist_ok=True)
                save_path = filedialog.asksaveasfilename(
                    title="Save synthetic scan",
                    initialdir=OUT,
                    defaultextension=".png",
                    filetypes=[("PNG image", "*.png")],
                )
                if not save_path:
                    return
            else:
                save_path = default_save_path()

        controls_row = {
            "requested_seeds": params[0],
            "preset": preset_var.get(),
            "touching_pct": int(touch_var.get()),
            "seed_brightness": f"{int(params[6][0])}-{int(params[6][1])}",
            "seed_scale": float(params[5]),
            "band_amp": float(params[3]),
            "max_cluster_size": int(cluster_var.get()),
        }

        set_busy(True)
        status_var.set("Generating synthetic scan...")

        def worker():
            try:
                rng = np.random.default_rng(seed)
                preset_name = controls_row["preset"] if controls_row["preset"] != "custom" else "custom_gui"
                img, truth = generate(preset_name, rng, bed, params)
                measured = None
                if save_path:
                    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                    Image.fromarray(img).save(save_path)
                    truth["file"] = os.path.basename(save_path)
                if analyze:
                    status = "Running SeedSizer on saved scan..."
                    root.after(0, lambda: status_var.set(status))
                    sys.path.insert(0, HERE)
                    import SeedSizer as S
                    with contextlib.redirect_stdout(io.StringIO()):
                        measured = S.Run(save_path)
                if save_path:
                    write_row = dict(truth)
                    write_row.update(controls_row)
                    write_row["file"] = os.path.basename(save_path)
                    if measured is not None:
                        got = int(measured["sscount"])
                        true_n = max(1, int(truth["n_seeds"]))
                        write_row["measured_count"] = got
                        write_row["count_error_pct"] = round(100.0 * (got - true_n) / true_n, 3)
                        write_row["processing_note"] = measured["ProcessingNote"]
                    _gui_write_truth(write_row)
                root.after(0, lambda: done(img, truth, save_path, measured, None))
            except Exception:
                err = traceback.format_exc()
                root.after(0, lambda err=err: done(None, None, None, None, err))

        def done(img, truth, path, measured, err):
            set_busy(False)
            if err:
                status_var.set("Generation failed")
                messagebox.showerror("Synthetic generator error", err)
                return
            last.update({"img": img, "truth": truth, "path": path})
            display_image(img)
            set_results(format_results(truth, controls_row, path, measured))
            if measured is not None:
                status_var.set("Saved and analyzed")
            elif path:
                status_var.set(f"Saved {os.path.basename(path)}")
            else:
                status_var.set("Preview generated")

        threading.Thread(target=worker, daemon=True).start()

    generate_button.config(command=lambda: run_job(save=False, analyze=False))
    save_button.config(command=lambda: run_job(save=True, analyze=False))
    analyze_button.config(command=lambda: run_job(save=True, analyze=True))
    choose_button.config(command=lambda: run_job(save=True, analyze=False, save_as=True))

    set_results("No scan generated yet.")
    root.mainloop()


# -------------------------------------------------------------------- check

def demo():
    """Self-check: the geometry and the truth bookkeeping must both hold."""
    m = ellipse(40.0, 25.0, 0.3)
    assert abs(m.sum() / (np.pi * 40 * 25) - 1) < 0.02, m.sum()

    small = (3000, 3000)                     # the full bed is 143 Mpx; too slow here
    rng = np.random.default_rng(0)
    img, t = generate("sparse", rng, small)
    assert t["n_seeds"] == 12, t
    assert 0.5 < t["mean_area_mm2"] / (SEED_AREA_MM2 * 0.60) < 1.6, t
    assert img.dtype == np.uint8 and img.shape[:2] == small, t

    assert img.ndim == 3 and img.shape[2] == 3, img.shape   # real scans are RGB
    med, mad = _mad(img[:, :, :3].mean(2))
    # Check the MEASURED background against the real scans (29.3-30.3), not
    # against BG_LEVEL: banding and vignette sit between the two on purpose.
    assert 28.0 <= med <= 32.0, med
    assert 2.5 < mad < 5.0, mad

    img, t = generate("sparse_clump", rng, small)
    assert t["n_touching"] > 0.5 * t["n_seeds"], t     # clusters really do cluster
    # The one that makes the clump test fair: touching seeds must not eat each
    # other's footprint, or area-based clump division fails for a fake reason.
    assert abs(t["overlap_pct"]) < 5.0, t

    try:
        generate("nope", rng, small)
    except KeyError:
        pass
    else:
        raise AssertionError("unknown preset should raise")
    print("ok")


if __name__ == "__main__":
    cmds = {"make": make, "screen": screen, "verify": verify, "batch": batch, "gui": gui, "demo": demo}
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd not in cmds:
        sys.exit(f"usage: python {os.path.basename(__file__)} "
                 f"{{{'|'.join(cmds)}}} [args]\n\n{__doc__}")

    def arg(a):
        try:
            return float(a) if "." in a else int(a)
        except ValueError:
            return a
    cmds[cmd](*[arg(a) for a in sys.argv[2:]])
