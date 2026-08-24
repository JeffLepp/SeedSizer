# SeedSizer — setup and what each file does

## Install

Python 3.11+ then:

    pip install numpy pandas scipy scikit-image imageio tifffile openpyxl pillow matplotlib seaborn

`tkinter` ships with the standard Python installer on Windows. If `import tkinter`
fails, reinstall Python with the "tcl/tk and IDLE" option ticked.

Check it works:

    python synth_test.py demo      # should print: ok
    python tiles.py                # should print usage

## The programs

| file | what it does |
|---|---|
| `SeedSizer.py` | the counter. `python SeedSizer.py` opens a folder queue picker and writes one CSV per folder. |
| `qc.py` | reconciles counts against balance weights and flags implausible TSM |
| `synth_test.py` | draws scans with known seed counts and scores SeedSizer against them |
| `tiles.py` | cuts real scans into windows for hand-counting, to check accuracy |
| `sweep_test.py` | earlier threshold sweep, kept for reference |

The folder queue window lets you add, remove, and reorder folders before running.
It processes them one at a time and writes each folder's normal `<folder>_data.csv`.

## Before trusting counts on a different scanner

The fallback threshold (`FALLBACK_THRESHOLD = 50` in SeedSizer.py) and the
constants in `synth_test.py` were measured on the 1200 PPI flatbed used for the
2025 scans: background ~29.7, MAD ~3.46, seeds peaking at 124-135. On different
hardware those numbers move, and a fallback above the seed brightness silently
loses seeds instead of failing loudly.

Run this on a folder of real scans from the new machine:

    python synth_test.py verify "C:\path\to\some\real\tifs"

Synthetic and real should agree on median, MAD and the >kMAD tail. If they do
not, the constants are stale and need re-measuring before the counts mean
anything.

## Known accuracy (as of 2026-08-13)

Measured against hand counts on six scans and 53 hand-counted tiles.

| condition | accuracy |
|---|---|
| healthy scans | exact (524 counted as 524; 1137 as 1136) |
| dense / touching | ~2% low |
| near-empty or failed pots | ~18% error, no blow-ups |

The failure this version fixes: Otsu picks a threshold by splitting a histogram
into two brightness modes. A nearly empty scan only has one mode - the
background - so Otsu splits background noise instead, calls half the bed "seed",
and the clump divider turns that single blob into tens of thousands of phantom
seeds. Pot_208 holds 12 seeds by hand count and was reported as 154,147.
SeedSizer now rejects any threshold selecting more than 15% of the bed, and says
so in the `ProcessingNote` column rather than silently rescuing the scan.

Still open: counts on heavily clumped beds run a few percent low, because the
reference seed size is the median of the scan's own objects and clumps drag that
median upward. Three separate fixes were tried and all failed held-out testing;
none are in the code.

## Regenerating the test data

`SynthScans/` and `TileCheck/` hold generated images, which are large and not in
git. Recreate them with:

    python synth_test.py make
    python tiles.py sample "path\to\scan.tif" 20 50

The `counts.csv` files under `TileCheck/` are hand counts and cannot be
regenerated - they are in git and should be kept.
