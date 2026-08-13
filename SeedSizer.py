from skimage.io             import imread
from skimage.filters        import threshold_otsu
from skimage.measure        import label, regionprops
from skimage.transform      import rescale
from skimage.measure        import regionprops_table
from skimage.morphology     import remove_small_objects
from pathlib                import Path
from scipy                  import ndimage as ndi
from skimage.feature        import peak_local_max
from skimage.segmentation   import watershed

import matplotlib.pyplot as plt
import imageio.v3 as iio
import seaborn as sns
import pandas as pd
import numpy as np
import tifffile
import openpyxl
import sys
import os
import gc

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # disable Pillow’s decompression bomb limit - don't bomb yourself :P
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import importlib.metadata
try:
    importlib.metadata.version("imageio")
except importlib.metadata.PackageNotFoundError:
    import imageio
    imageio.__version__ = "3.0.0"  # dummy fallback for frozen apps


PPI = 1200                                                                      # Pixels per inch
PP_SQMM = (PPI / 25.4) ** 2                                                     # Pixels per square millimeter
PX_PER_MM = PPI / 25.4                                                          # pixels per millimeter
FILTER = 0.1                                                                    # This is basically used saying we won't accept anything < .1 mm^2 in size
MINSIZE = .5                                                                    # This is the minimum size of a seed we want to consider, in this case 50% of the median area   

MIN_SEED_AREA_MM2 = 0.12                                                        # Absolute lower bound; keeps dust from defining "median seed"
MIN_REFERENCE_SEED_AREA_MM2 = 0.20                                              # Floor used for clump estimates when scans are noisy
MAX_SINGLE_SEED_AREA_MM2 = 6.0
MAX_CLUMP_AREA_MM2 = 200.0
MAX_SINGLE_ASPECT_RATIO = 5.5
MAX_CLUMP_ASPECT_RATIO = 9.0
MIN_SOLIDITY = 0.45
CLUMP_FACTOR = 1.9

DEGENERATE_FG = 0.15                                                            # >15% of the bed being seed is physically impossible for one layer
FALLBACK_THRESHOLD = 62.0                                                       # sits in the empty gap between background (~30) and seed (70-135)

#################################################################################################################################################################################################
#
# Though to make this more user friendly, I made adjustable parameters above to match your image requirements. 
#
#       PPI is the pixels per inch, which is 1200 for the .tif images used to develope this.
#       PP_SQMM is the pixels per square millimeter, which is calculated from PPI, you shouldn't have to change this even if your PPI is different. If you want metric distance, change the equation accordingly.
#       FILTER is the minimum size of a object (in my case a seed) you want to even accept. This is used such that it removed objects smaller than a 1/10 mm^2.
#       MINSIZE is the minimum size of a object you want to include in calculations. This is used such that it removed objects smaller than 4/10 the median size of all objects.
#
#################################################################################################################################################################################################


def _as_grayscale_float(raw_image):
    if raw_image.ndim == 2:
        return raw_image.astype(np.float32)

    return np.mean(raw_image[:, :, :3], axis=2).astype(np.float32)


def _quality_note(parts):
    return " ".join(part for part in parts if part)


def _empty_result(path, processing_note, raw_object_count, rejected_object_count=None):
    if rejected_object_count is None:
        rejected_object_count = raw_object_count

    print(f"Filepath: {path}")
    print("Total number of filtered seeds: 0")
    print(f"Average seed size: unavailable ({processing_note})")
    print()

    return {
        "fileName": path.name,
        "objectNumber": "",
        "Area": "",
        "StdArea": "",
        "Length": "",
        "StdLength": "",
        "Width": "",
        "StdWidth": "",
        "Eccentricity": "",
        "StdEccentricity": "",
        "sscount": 0,
        "AvgSizeOfOneSeed": "",
        "ProcessingNote": processing_note,
        "RawObjectCount": int(raw_object_count),
        "AcceptedObjectCount": 0,
        "RejectedObjectCount": int(rejected_object_count),
        "ReferenceSeedArea": "",
    }


def _degenerate_prefix(was_degenerate):
    """Keep the fallback visible on the early-exit paths too, so a scan is never
    silently rescued without it showing up in the output."""
    return f"Otsu was degenerate; used fixed threshold {FALLBACK_THRESHOLD:.0f}. " if was_degenerate else ""


def Run(filename):

    ### Image Manipulation ###

    path = Path(filename).resolve()
    raw_image = iio.imread(path)
    grayscale_image = _as_grayscale_float(raw_image)
    del raw_image
    gc.collect() 

    threshold_value = threshold_otsu(grayscale_image)

    # Otsu splits a histogram into two modes. A nearly empty scan only has one -
    # the background - so Otsu splits background noise instead and calls half the
    # bed "seed"; the clump divider then turns that single blob into tens of
    # thousands of phantom seeds (Pot_208: 12 real seeds reported as 154,147).
    # No seed layer can cover 15% of the bed, so treat that as the tell and cut
    # in the empty gap between background and seed instead. Verified to leave
    # healthy scans bit-for-bit unchanged.
    otsu_degenerate = float((grayscale_image > threshold_value).mean()) > DEGENERATE_FG
    if otsu_degenerate:
        threshold_value = FALLBACK_THRESHOLD

    binary_image = grayscale_image > threshold_value
    binary_clean = remove_small_objects(binary_image, min_size=int(PP_SQMM * FILTER))
    labeled_image = label(binary_clean)


    ### Image Analysis ###

    binary_seed = regionprops_table(
        labeled_image,
        properties=[
            "label",
            "area",
            "eccentricity",
            "solidity",
            "major_axis_length",
            "minor_axis_length",
        ],
    )        
                                                                                        # ^^ Collection of area's of the connected components (collection of adjascent pixels labeled 1, which make up the seed)
    df = pd.DataFrame(binary_seed)                                                      # This turns our dictionary of connected components into a pandas dataframe
    raw_object_count = len(df)

    if df.empty:
        return _empty_result(path, _degenerate_prefix(otsu_degenerate)
                             + "No seed-like objects found after thresholding.", raw_object_count, 0)

    df["area_mm2"] = df["area"] / PP_SQMM                                               # Convert these connected components to mm^2 since we know pixel is 1/1200 of an inch
    df["aspect_ratio"] = df["major_axis_length"] / df["minor_axis_length"]
    df["length_mm"] = df["major_axis_length"] / PX_PER_MM
    df["width_mm"] = df["minor_axis_length"] / PX_PER_MM
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    if df.empty:
        return _empty_result(path, _degenerate_prefix(otsu_degenerate)
                             + "All detected objects had invalid shape measurements.", raw_object_count)

    ### Statistical Analysis ###

    plausible_single = df[
        (df["area_mm2"].between(MIN_SEED_AREA_MM2, MAX_SINGLE_SEED_AREA_MM2))
        & (df["aspect_ratio"] <= MAX_SINGLE_ASPECT_RATIO)
        & (df["solidity"] >= MIN_SOLIDITY)
    ]
    if plausible_single.empty:
        reference_seed_area = max(float(df["area_mm2"].median()), MIN_REFERENCE_SEED_AREA_MM2)
    else:
        reference_seed_area = max(float(plausible_single["area_mm2"].median()), MIN_REFERENCE_SEED_AREA_MM2)

    min_area_mm2 = max(MIN_SEED_AREA_MM2, MINSIZE * reference_seed_area)
    shape_ok = (
        (df["aspect_ratio"] <= MAX_SINGLE_ASPECT_RATIO)
        & (df["solidity"] >= MIN_SOLIDITY)
    )
    clump_shape_ok = (
        (df["aspect_ratio"] <= MAX_CLUMP_ASPECT_RATIO)
        & (df["solidity"] >= MIN_SOLIDITY * 0.65)
        & (df["area_mm2"] <= MAX_CLUMP_AREA_MM2)
    )
    df_filtered = df[
        (df["area_mm2"] >= min_area_mm2)
        & (shape_ok | ((df["area_mm2"] > CLUMP_FACTOR * reference_seed_area) & clump_shape_ok))
    ]

    clumps = df_filtered[df_filtered["area_mm2"] > CLUMP_FACTOR * reference_seed_area].copy()
    clumps["clump_size"] = np.maximum(2, (clumps["area_mm2"] / reference_seed_area).round().astype(int))

    size_clumps = clumps["clump_size"].sum()                                            # Counting number of seeds in clumps
    size_singles = len(df_filtered[df_filtered["area_mm2"] <= CLUMP_FACTOR * reference_seed_area])
    total_size = size_clumps + size_singles                                             # Aggregate seed count                                      
    total_area = df_filtered["area_mm2"].sum()

    single_seed_areas = df_filtered[df_filtered["area_mm2"] <= CLUMP_FACTOR * reference_seed_area]["area_mm2"]
    mean_beta = single_seed_areas.mean()
    quality_notes = []
    if otsu_degenerate:
        quality_notes.append(
            f"Otsu was degenerate (>{DEGENERATE_FG:.0%} foreground); used fixed "
            f"threshold {FALLBACK_THRESHOLD:.0f}. Scan is nearly empty - verify against weight."
        )
    if pd.isna(mean_beta):
        if df_filtered.empty:
            quality_notes.append("No filtered seed objects found; average single seed size unavailable.")
        else:
            quality_notes.append("No single seed objects found; detected objects may all be clumps.")
    if raw_object_count > 0 and len(df_filtered) / raw_object_count < 0.25:
        quality_notes.append("Most thresholded objects were rejected as dust/artifacts.")
    if len(df_filtered) >= 20 and reference_seed_area <= MIN_REFERENCE_SEED_AREA_MM2 * 1.05:
        quality_notes.append("Reference seed area hit the lower safety bound; inspect scan quality.")
    processing_note = _quality_note(quality_notes)

    length_mm = df_filtered["length_mm"]
    width_mm  = df_filtered["width_mm"]

    # --- Requested stats / column names ---
    area_mean = df_filtered["area_mm2"].mean()
    area_std  = df_filtered["area_mm2"].std()

    len_mean  = length_mm.mean()
    len_std   = length_mm.std()

    wid_mean  = width_mm.mean()
    wid_std   = width_mm.std()

    ecc_mean  = df_filtered["eccentricity"].mean()
    ecc_std   = df_filtered["eccentricity"].std()

    ### Output ###
    print(f"Filepath: {path}")
    print(f"Total number of filtered seeds: {total_size}")
    if pd.isna(mean_beta):
        print(f"Average seed size: unavailable ({processing_note})")
    else:
        print(f"Average seed size: {mean_beta:.3f} mm²")
        if processing_note:
            print(f"Processing note: {processing_note}")
    print()

    return {
        "fileName": path.name,
        "objectNumber": "",             # per-image summary => leave blank; see note below
        "Area": float(area_mean),       # mean area (mm^2)
        "StdArea": float(area_std),     # std dev of area (mm^2)
        "Length": float(len_mean),      # mean major axis (mm)
        "StdLength": float(len_std),    # std dev major axis (mm)
        "Width": float(wid_mean),       # mean minor axis (mm)
        "StdWidth": float(wid_std),     # std dev minor axis (mm)
        "Eccentricity": float(ecc_mean),# mean eccentricity (0-1) - correlates to degree of roundness vs oval
        "StdEccentricity": float(ecc_std),
        "sscount": int(total_size),     # clump-aware seed count
        "AvgSizeOfOneSeed": "" if pd.isna(mean_beta) else float(mean_beta),
        "ProcessingNote": processing_note,
        "RawObjectCount": int(raw_object_count),
        "AcceptedObjectCount": int(len(df_filtered)),
        "RejectedObjectCount": int(raw_object_count - len(df_filtered)),
        "ReferenceSeedArea": float(reference_seed_area),
    }

# This is where we cycle through each image in the folder that was passed by the user
def Cycle(folder="Data"):
    folder_path = Path(folder)
    tif_files = list(folder_path.glob("*.tif")) + list(folder_path.glob("*.tiff"))
    output_csv = Path(folder_path.parent) / f"{folder_path.name}_data.csv"

    # --- Create small progress window ---
    progress_win = tk.Toplevel()
    progress_win.title("SeedSizer Progress")
    progress_win.geometry("400x120")
    progress_win.resizable(False, False)

    label_status = tk.Label(progress_win, text="Processing TIFF images...", font=("Segoe UI", 11))
    label_status.pack(pady=10)

    label_file = tk.Label(progress_win, text="", font=("Segoe UI", 9))
    label_file.pack()

    progress_bar = ttk.Progressbar(progress_win, length=320, mode="determinate")
    progress_bar.pack(pady=10)
    progress_bar["maximum"] = len(tif_files)

    progress_win.update()

    result = []
    for i, tif_file in enumerate(tif_files):

        label_file.config(text=f"→ {tif_file.name}")
        progress_bar["value"] = i
        progress_win.update()

        stats = Run(tif_file)

        df_row = pd.DataFrame([stats]) 
        file_exists = output_csv.exists()

        # append cycle data to output CSV
        df_row.to_csv(output_csv, mode="a", header=not file_exists, index=False)
        print(f"Added {tif_file.name} to {output_csv.name}", flush=True)

    progress_bar["value"] = len(tif_files)
    label_status.config(text=f"Complete — saved to {output_csv.name}")
    label_file.config(text="")
    progress_win.update()


# Run SeedSizer.py as a standalone program
# You can also convert to a .exe using pyinstaller and run on any device without Python installed

# Note folder query may take a few seconds due to filedialog performance on some systems

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # hide main window
    folder = filedialog.askdirectory(title="Select folder containing .TIFF images")

    if not folder:
        print("No folder selected. Exiting.")
        sys.exit(0)

    Cycle(folder)
