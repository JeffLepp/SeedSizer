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
import contextlib
import hashlib
import io
import re
import sys
import os
import gc
import threading
import time

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

from PIL import Image, ImageTk, ImageDraw
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
FALLBACK_THRESHOLD = 50.0                                                       # fallback for Otsu-degenerate scans; keeps dim seed bodies intact
MIN_PLAUSIBLE_DPI = 300
MAX_PLAUSIBLE_DPI = 2400
MAX_DPI_AXIS_MISMATCH = 0.02
METADATA_DPI_WARNING = 0.02

OVERLAY_MAX_SIDE = 1800
OVERLAY_COLORS = {
    "single": (46, 204, 113, 110),
    "clump": (255, 193, 7, 135),
    "rejected": (231, 76, 60, 115),
    "single_outline": (0, 128, 72, 255),
    "clump_outline": (176, 112, 0, 255),
    "rejected_outline": (150, 28, 20, 255),
}

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


def _ratio_to_float(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        den = float(value[1])
        return None if den == 0 else float(value[0]) / den
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        den = float(value.denominator)
        return None if den == 0 else float(value.numerator) / den
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolution_unit_to_inches(unit):
    if unit is None:
        return 1.0
    text = str(unit).lower()
    if "centimeter" in text or text in {"3", "resolutionunit.centimeter"}:
        return 2.54
    if "inch" in text or text in {"2", "resolutionunit.inch"}:
        return 1.0
    return None


def _read_metadata_dpi(path):
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        try:
            with tifffile.TiffFile(path) as tif:
                tags = tif.pages[0].tags
                x = _ratio_to_float(tags["XResolution"].value if "XResolution" in tags else None)
                y = _ratio_to_float(tags["YResolution"].value if "YResolution" in tags else None)
                unit = _resolution_unit_to_inches(tags["ResolutionUnit"].value if "ResolutionUnit" in tags else None)
                if x and y and unit:
                    return x * unit, y * unit, "TIFF metadata"
        except Exception:
            pass

    try:
        with Image.open(path) as img:
            dpi = img.info.get("dpi")
            if dpi and len(dpi) >= 2:
                x = _ratio_to_float(dpi[0])
                y = _ratio_to_float(dpi[1])
                if x and y:
                    return x, y, "image metadata"
    except Exception:
        pass

    return None, None, None


def _format_dpi(value):
    return f"{float(value):.0f}" if abs(float(value) - round(float(value))) < 0.05 else f"{float(value):.1f}"


def _calibration_from_settings(path, calibration=None):
    calibration = calibration or {}
    mode = calibration.get("mode", "auto")
    default_dpi = float(calibration.get("default_dpi", PPI) or PPI)
    manual_dpi = float(calibration.get("manual_dpi", default_dpi) or default_dpi)
    ppi = default_dpi
    mode_label = mode
    notes = []

    metadata_x, metadata_y, metadata_source = _read_metadata_dpi(path)
    metadata_valid = False
    metadata_ppi = None
    if metadata_x and metadata_y:
        metadata_ppi = (metadata_x + metadata_y) / 2.0
        axis_mismatch = abs(metadata_x - metadata_y) / max(metadata_ppi, 1.0)
        metadata_valid = (
            MIN_PLAUSIBLE_DPI <= metadata_ppi <= MAX_PLAUSIBLE_DPI
            and axis_mismatch <= MAX_DPI_AXIS_MISMATCH
        )
        if axis_mismatch > MAX_DPI_AXIS_MISMATCH:
            notes.append(
                f"Image DPI axes disagree ({_format_dpi(metadata_x)} x {_format_dpi(metadata_y)});"
                f" using {_format_dpi(default_dpi)} DPI default."
            )
        elif not (MIN_PLAUSIBLE_DPI <= metadata_ppi <= MAX_PLAUSIBLE_DPI):
            notes.append(
                f"Image DPI metadata looks implausible ({_format_dpi(metadata_ppi)});"
                f" using {_format_dpi(default_dpi)} DPI default."
            )

    if mode == "manual":
        if MIN_PLAUSIBLE_DPI <= manual_dpi <= MAX_PLAUSIBLE_DPI:
            ppi = manual_dpi
            mode_label = "manual"
            if metadata_valid and abs(metadata_ppi - ppi) / ppi > METADATA_DPI_WARNING:
                notes.append(
                    f"Manual DPI {_format_dpi(ppi)} overrides {metadata_source}"
                    f" {_format_dpi(metadata_ppi)}."
                )
        else:
            notes.append(
                f"Manual DPI {_format_dpi(manual_dpi)} looks implausible;"
                f" using {_format_dpi(default_dpi)} DPI default."
            )
    elif mode == "fixed":
        ppi = default_dpi
        mode_label = "fixed"
        if metadata_valid and abs(metadata_ppi - ppi) / ppi > METADATA_DPI_WARNING:
            notes.append(
                f"Image metadata says {_format_dpi(metadata_ppi)} DPI;"
                f" fixed setting uses {_format_dpi(ppi)} DPI."
            )
    else:
        mode_label = "auto"
        if metadata_valid:
            ppi = metadata_ppi
            if abs(ppi - default_dpi) / default_dpi > METADATA_DPI_WARNING:
                notes.append(
                    f"Using {metadata_source} DPI {_format_dpi(ppi)} instead of"
                    f" default {_format_dpi(default_dpi)}."
                )
        elif not notes:
            notes.append(
                f"No reliable image DPI metadata found; using {_format_dpi(default_dpi)} DPI default."
            )

    pp_sqmm = (ppi / 25.4) ** 2
    px_per_mm = ppi / 25.4
    return {
        "mode": mode_label,
        "ppi": float(ppi),
        "pp_sqmm": float(pp_sqmm),
        "px_per_mm": float(px_per_mm),
        "note": _quality_note(notes),
    }


def _calibration_columns(calibration_info):
    return {
        "CalibrationMode": calibration_info["mode"],
        "CalibrationPPI": float(calibration_info["ppi"]),
        "CalibrationNote": calibration_info["note"],
    }


def _empty_result(path, processing_note, raw_object_count, rejected_object_count=None, calibration_info=None):
    if rejected_object_count is None:
        rejected_object_count = raw_object_count
    calibration_info = calibration_info or _calibration_from_settings(path)
    processing_note = _quality_note([processing_note, calibration_info["note"]])

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
        **_calibration_columns(calibration_info),
    }


def _degenerate_prefix(was_degenerate):
    """Keep the fallback visible on the early-exit paths too, so a scan is never
    silently rescued without it showing up in the output."""
    return f"Otsu was degenerate; used fixed threshold {FALLBACK_THRESHOLD:.0f}. " if was_degenerate else ""


def _diagnostics_payload(path, grayscale_image, labeled_image, df_all, df_filtered, clumps,
                         threshold_value, otsu_degenerate, result):
    accepted_labels = set(df_filtered["label"].astype(int)) if "label" in df_filtered else set()
    clump_labels = set(clumps["label"].astype(int)) if "label" in clumps else set()
    all_labels = set(df_all["label"].astype(int)) if "label" in df_all else set()
    return {
        "path": path,
        "grayscale": grayscale_image,
        "labeled": labeled_image,
        "df_all": df_all.copy(),
        "df_filtered": df_filtered.copy(),
        "clumps": clumps.copy(),
        "accepted_labels": accepted_labels,
        "clump_labels": clump_labels,
        "rejected_labels": all_labels - accepted_labels,
        "threshold": float(threshold_value),
        "otsu_degenerate": bool(otsu_degenerate),
        "result": result,
    }


def _analyze_scan(filename, calibration=None, include_diagnostics=False):

    ### Image Manipulation ###

    path = Path(filename).resolve()
    calibration_info = _calibration_from_settings(path, calibration)
    pp_sqmm = calibration_info["pp_sqmm"]
    px_per_mm = calibration_info["px_per_mm"]
    raw_image = iio.imread(path)
    grayscale_image = _as_grayscale_float(raw_image)
    del raw_image
    gc.collect() 

    threshold_value = threshold_otsu(grayscale_image)

    # Otsu splits a histogram into two modes. A nearly empty scan only has one -
    # the background - so Otsu splits background noise instead and calls half the
    # bed "seed"; the clump divider then turns that single blob into tens of
    # thousands of phantom seeds (Pot_208: 12 real seeds reported as 154,147).
    # No seed layer can cover 15% of the bed, so treat that as the tell and use
    # a fixed threshold that keeps dim seed bodies intact instead of isolating
    # only bright cores (Pot_163: ~16 real seeds reported as 32 at threshold 62).
    otsu_degenerate = float((grayscale_image > threshold_value).mean()) > DEGENERATE_FG
    if otsu_degenerate:
        threshold_value = FALLBACK_THRESHOLD

    binary_image = grayscale_image > threshold_value
    binary_clean = remove_small_objects(binary_image, min_size=int(pp_sqmm * FILTER))
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
            "centroid",
        ],
    )        
                                                                                        # ^^ Collection of area's of the connected components (collection of adjascent pixels labeled 1, which make up the seed)
    df = pd.DataFrame(binary_seed)                                                      # This turns our dictionary of connected components into a pandas dataframe
    df_all = df.copy()
    raw_object_count = len(df)

    if df.empty:
        result = _empty_result(path, _degenerate_prefix(otsu_degenerate)
                               + "No seed-like objects found after thresholding.", raw_object_count, 0, calibration_info)
        diagnostics = _diagnostics_payload(
            path, grayscale_image, labeled_image, df_all, df, pd.DataFrame(),
            threshold_value, otsu_degenerate, result,
        ) if include_diagnostics else None
        return result, diagnostics

    df["area_mm2"] = df["area"] / pp_sqmm                                               # Convert connected components to mm^2 using the active scan calibration
    df["aspect_ratio"] = df["major_axis_length"] / df["minor_axis_length"]
    df["length_mm"] = df["major_axis_length"] / px_per_mm
    df["width_mm"] = df["minor_axis_length"] / px_per_mm
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    if df.empty:
        result = _empty_result(path, _degenerate_prefix(otsu_degenerate)
                               + "All detected objects had invalid shape measurements.", raw_object_count,
                               calibration_info=calibration_info)
        diagnostics = _diagnostics_payload(
            path, grayscale_image, labeled_image, df_all, df, pd.DataFrame(),
            threshold_value, otsu_degenerate, result,
        ) if include_diagnostics else None
        return result, diagnostics

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
            f"threshold {FALLBACK_THRESHOLD:.0f}. Threshold metadata looked unreliable; inspect scan density, lighting, and background."
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
    if calibration_info["note"]:
        quality_notes.append(f"Calibration check: {calibration_info['note']}")
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

    result = {
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
        **_calibration_columns(calibration_info),
    }
    diagnostics = _diagnostics_payload(
        path, grayscale_image, labeled_image, df_all, df_filtered, clumps,
        threshold_value, otsu_degenerate, result,
    ) if include_diagnostics else None
    return result, diagnostics


def Run(filename, calibration=None):
    result, _diagnostics = _analyze_scan(filename, calibration=calibration, include_diagnostics=False)
    return result


def _scale_to_uint8(grayscale_image):
    finite = grayscale_image[np.isfinite(grayscale_image)]
    if finite.size == 0:
        return np.zeros(grayscale_image.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(grayscale_image.shape, dtype=np.uint8)
    scaled = (np.clip(grayscale_image, low, high) - low) * (255.0 / (high - low))
    return scaled.astype(np.uint8)


def _boundary_mask(mask):
    if not np.any(mask):
        return mask
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return mask ^ eroded


def _make_overlay_image(diagnostics, max_side=OVERLAY_MAX_SIDE):
    grayscale = diagnostics["grayscale"]
    labeled_image = diagnostics["labeled"]
    accepted_labels = diagnostics["accepted_labels"]
    clump_labels = diagnostics["clump_labels"]
    rejected_labels = diagnostics["rejected_labels"]
    single_labels = accepted_labels - clump_labels

    category = np.zeros(labeled_image.shape, dtype=np.uint8)
    if rejected_labels:
        category[np.isin(labeled_image, list(rejected_labels))] = 3
    if single_labels:
        category[np.isin(labeled_image, list(single_labels))] = 1
    if clump_labels:
        category[np.isin(labeled_image, list(clump_labels))] = 2

    boundary = np.zeros(labeled_image.shape, dtype=np.uint8)
    for value in (1, 2, 3):
        mask = category == value
        if np.any(mask):
            boundary[_boundary_mask(mask)] = value

    bg = Image.fromarray(_scale_to_uint8(grayscale)).convert("RGB")
    original_size = bg.size
    bg.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    display_size = bg.size

    category_img = Image.fromarray(category).resize(display_size, Image.Resampling.NEAREST)
    boundary_img = Image.fromarray(boundary).resize(display_size, Image.Resampling.NEAREST)
    category_small = np.array(category_img)
    boundary_small = np.array(boundary_img)

    overlay = Image.new("RGBA", display_size, (0, 0, 0, 0))
    overlay_pixels = np.array(overlay)
    color_map = {
        1: OVERLAY_COLORS["single"],
        2: OVERLAY_COLORS["clump"],
        3: OVERLAY_COLORS["rejected"],
    }
    outline_map = {
        1: OVERLAY_COLORS["single_outline"],
        2: OVERLAY_COLORS["clump_outline"],
        3: OVERLAY_COLORS["rejected_outline"],
    }
    for value, color in color_map.items():
        overlay_pixels[category_small == value] = color
    for value, color in outline_map.items():
        overlay_pixels[boundary_small == value] = color

    combined = Image.alpha_composite(bg.convert("RGBA"), Image.fromarray(overlay_pixels, mode="RGBA"))
    draw = ImageDraw.Draw(combined)
    scale_x = display_size[0] / max(1, original_size[0])
    scale_y = display_size[1] / max(1, original_size[1])
    clumps = diagnostics["clumps"]
    if not clumps.empty:
        for _, row in clumps.iterrows():
            x = float(row.get("centroid-1", 0)) * scale_x
            y = float(row.get("centroid-0", 0)) * scale_y
            text = str(int(row.get("clump_size", 2)))
            draw.text((x + 4, y + 4), text, fill=(255, 255, 255, 255))

    return combined.convert("RGB")


def CreateOverlay(filename, calibration=None, output_path=None, max_side=OVERLAY_MAX_SIDE):
    result, diagnostics = _analyze_scan(filename, calibration=calibration, include_diagnostics=True)
    overlay = _make_overlay_image(diagnostics, max_side=max_side)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(output_path)
    return overlay, result


# This is where we cycle through each image in the folder that was passed by the user
def _find_tif_files(folder_path):
    return sorted(
        path for path in folder_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )


def _safe_filename_part(value):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return safe or "scan"


def _folder_output_tag(folder_path):
    resolved = str(Path(folder_path).resolve())
    return hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:8]


def _default_output_dir(folder_path):
    return Path(folder_path).parent


def _output_csv_path(folder_path, output_dir=None, reserved_paths=None):
    folder_path = Path(folder_path)
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = _default_output_dir(folder_path)

    try:
        custom_output = output_dir.resolve() != _default_output_dir(folder_path).resolve()
    except OSError:
        custom_output = True

    stem = f"{_safe_filename_part(folder_path.name)}_data"
    if custom_output:
        stem = f"{stem}_{_folder_output_tag(folder_path)}"

    candidate = output_dir / f"{stem}.csv"
    if reserved_paths is not None:
        index = 2
        while candidate.resolve() in reserved_paths:
            candidate = output_dir / f"{stem}_{index}.csv"
            index += 1
        reserved_paths.add(candidate.resolve())
    return candidate


def _create_progress_window(total_files, root=None):
    try:
        if root is None:
            root = tk.Tk()
            root.withdraw()

        progress_win = tk.Toplevel(root)
        progress_win.title("SeedSizer Progress")
        progress_win.geometry("400x120")
        progress_win.resizable(False, False)

        label_status = tk.Label(progress_win, text="Processing TIFF images...", font=("Segoe UI", 11))
        label_status.pack(pady=10)

        label_file = tk.Label(progress_win, text="", font=("Segoe UI", 9))
        label_file.pack()

        progress_bar = ttk.Progressbar(progress_win, length=320, mode="determinate")
        progress_bar.pack(pady=10)
        progress_bar["maximum"] = total_files

        progress_win.update()
        return progress_win, label_status, label_file, progress_bar
    except tk.TclError as exc:
        print(f"Progress window unavailable ({exc}); continuing in console.", flush=True)
        return None


def Cycle(folder="Data", root=None, close_progress=False, progress_callback=None,
          output_dir=None, reserved_outputs=None, calibration=None):
    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"Selected folder does not exist: {folder_path}")
        if progress_callback:
            progress_callback("folder_error", folder_path=folder_path, message="Folder does not exist")
        return {"folder": folder_path, "output_csv": None, "rows": []}

    tif_files = _find_tif_files(folder_path)
    output_csv = _output_csv_path(folder_path, output_dir, reserved_outputs)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(
            "folder_start",
            folder_path=folder_path,
            output_csv=output_csv,
            file_index=0,
            total_files=len(tif_files),
        )
        progress = None
    else:
        progress = _create_progress_window(len(tif_files), root)

    if not tif_files:
        print(f"No .tif or .tiff files found in {folder_path}")
        if progress_callback:
            progress_callback(
                "folder_empty",
                folder_path=folder_path,
                output_csv=output_csv,
                file_index=0,
                total_files=0,
                message="No TIFF images found",
            )
        if progress:
            progress_win, label_status, label_file, progress_bar = progress
            label_status.config(text="No TIFF images found")
            progress_win.update()
            if close_progress:
                progress_win.destroy()
        return {"folder": folder_path, "output_csv": output_csv, "rows": []}

    result = []
    for i, tif_file in enumerate(tif_files):

        if progress_callback:
            progress_callback(
                "file_start",
                folder_path=folder_path,
                output_csv=output_csv,
                file_index=i,
                total_files=len(tif_files),
                file_name=tif_file.name,
            )
        elif progress:
            progress_win, label_status, label_file, progress_bar = progress
            label_file.config(text=f"Processing {tif_file.name}")
            progress_bar["value"] = i
            progress_win.update()

        stats = Run(tif_file, calibration=calibration)
        result.append({"path": tif_file, "stats": stats, "output_csv": output_csv})

        df_row = pd.DataFrame([stats]) 
        file_exists = output_csv.exists()

        # append cycle data to output CSV
        df_row.to_csv(output_csv, mode="a", header=not file_exists, index=False)
        print(f"Added {tif_file.name} to {output_csv.name}", flush=True)
        if progress_callback:
            progress_callback(
                "file_done",
                folder_path=folder_path,
                output_csv=output_csv,
                file_index=i + 1,
                total_files=len(tif_files),
                file_name=tif_file.name,
            )

    if progress_callback:
        progress_callback(
            "folder_done",
            folder_path=folder_path,
            output_csv=output_csv,
            file_index=len(tif_files),
            total_files=len(tif_files),
        )
    elif progress:
        progress_win, label_status, label_file, progress_bar = progress
        progress_bar["value"] = len(tif_files)
        label_status.config(text=f"Complete - saved to {output_csv.name}")
        label_file.config(text="")
        progress_win.update()
        if close_progress:
            progress_win.destroy()

    return {"folder": folder_path, "output_csv": output_csv, "rows": result}


def CycleQueue(folders, root=None, output_dirs=None, calibration=None):
    folders = [Path(folder) for folder in folders if folder]
    output_dirs = list(output_dirs or [None] * len(folders))
    reserved_outputs = set()
    total_folders = len(folders)
    queue_results = []
    for index, folder in enumerate(folders, start=1):
        print(f"Starting folder {index}/{total_folders}: {folder}", flush=True)
        output_dir = output_dirs[index - 1] if index - 1 < len(output_dirs) else None
        summary = Cycle(folder, root, close_progress=index < total_folders,
                        output_dir=output_dir, reserved_outputs=reserved_outputs, calibration=calibration)
        if summary:
            queue_results.extend(summary["rows"])
    return queue_results


def _short_note(text, limit=90):
    text = str(text or "")
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _show_overlay_window(parent, scan_path, calibration=None, default_dir=None):
    scan_path = Path(scan_path)
    overlay_win = tk.Toplevel(parent)
    overlay_win.title(f"SeedSizer Overlay - {scan_path.name}")
    overlay_win.geometry("1120x820")
    overlay_win.minsize(840, 620)
    overlay_win.configure(bg="#f4f7f5")

    top = tk.Frame(overlay_win, bg="#f4f7f5")
    top.pack(fill="x", padx=14, pady=(12, 8))
    tk.Label(
        top,
        text=scan_path.name,
        bg="#f4f7f5",
        fg="#24322e",
        font=("Segoe UI", 12, "bold"),
        anchor="w",
    ).pack(fill="x")

    legend = tk.Frame(overlay_win, bg="#f4f7f5")
    legend.pack(fill="x", padx=14, pady=(0, 8))
    for label_text, color in (
        ("Accepted single seeds", "#2ecc71"),
        ("Counted clumps", "#ffc107"),
        ("Rejected objects", "#e74c3c"),
    ):
        chip = tk.Canvas(legend, width=16, height=16, bg="#f4f7f5", highlightthickness=0)
        chip.create_rectangle(2, 2, 14, 14, fill=color, outline="")
        chip.pack(side="left", padx=(0, 5))
        tk.Label(
            legend,
            text=label_text,
            bg="#f4f7f5",
            fg="#34423d",
            font=("Segoe UI", 9),
        ).pack(side="left", padx=(0, 16))

    image_frame = tk.Frame(overlay_win, bg="#101820", highlightthickness=1, highlightbackground="#20343b")
    image_frame.pack(fill="both", expand=True, padx=14, pady=(0, 10))
    image_frame.columnconfigure(0, weight=1)
    image_frame.rowconfigure(0, weight=1)

    image_label = tk.Label(
        image_frame,
        text="Building overlay...",
        bg="#101820",
        fg="#c7d8d2",
        font=("Segoe UI", 12),
    )
    image_label.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

    footer = tk.Frame(overlay_win, bg="#f4f7f5")
    footer.pack(fill="x", padx=14, pady=(0, 12))
    status = tk.StringVar(value="Thresholding scan and drawing detected objects...")
    tk.Label(
        footer,
        textvariable=status,
        bg="#f4f7f5",
        fg="#52615c",
        font=("Segoe UI", 9),
        anchor="w",
    ).pack(side="left", fill="x", expand=True)

    save_button = ttk.Button(footer, text="Save Overlay", state=tk.DISABLED)
    save_button.pack(side="right")

    last = {"image": None, "photo": None}

    def display_overlay(overlay, result, err):
        if err is not None:
            status.set("Overlay failed")
            image_label.config(text="Could not build overlay.")
            messagebox.showerror("SeedSizer Overlay", str(err), parent=overlay_win)
            return

        display = overlay.copy()
        display.thumbnail((1040, 650), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display)
        image_label.config(image=photo, text="")
        image_label.image = photo
        last["image"] = overlay
        last["photo"] = photo
        status.set(
            f"Count {result['sscount']} | accepted {result['AcceptedObjectCount']} | "
            f"rejected {result['RejectedObjectCount']} | raw {result['RawObjectCount']}"
        )
        save_button.config(state=tk.NORMAL)

    def save_overlay():
        if last["image"] is None:
            return
        folder = Path(default_dir) if default_dir else scan_path.parent / "overlays"
        folder.mkdir(parents=True, exist_ok=True)
        filename = filedialog.asksaveasfilename(
            parent=overlay_win,
            title="Save overlay image",
            initialdir=str(folder),
            initialfile=f"{scan_path.stem}_overlay.png",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("All files", "*.*")],
        )
        if filename:
            last["image"].save(filename)
            status.set(f"Saved overlay to {filename}")

    save_button.config(command=save_overlay)

    def worker():
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                overlay, result = CreateOverlay(scan_path, calibration=calibration)
            overlay_win.after(0, lambda: display_overlay(overlay, result, None))
        except Exception as exc:
            overlay_win.after(0, lambda exc=exc: display_overlay(None, None, exc))

    threading.Thread(target=worker, daemon=True).start()


def _show_results_window(parent, rows, title="SeedSizer Results", output_csv=None, calibration=None):
    rows = list(rows or [])
    if not rows:
        messagebox.showinfo("SeedSizer Results", "No scan results to display.", parent=parent)
        return

    result_win = tk.Toplevel(parent)
    result_win.title(title)
    result_win.geometry("1120x560")
    result_win.minsize(860, 420)
    result_win.configure(bg="#f4f7f5")

    header = tk.Frame(result_win, bg="#f4f7f5")
    header.pack(fill="x", padx=14, pady=(12, 8))
    tk.Label(
        header,
        text=title,
        bg="#f4f7f5",
        fg="#24322e",
        font=("Segoe UI", 14, "bold"),
        anchor="w",
    ).pack(fill="x")
    detail = f"Saved to {output_csv}" if output_csv else "Select a row to inspect detection highlighting."
    tk.Label(
        header,
        text=detail,
        bg="#f4f7f5",
        fg="#52615c",
        font=("Segoe UI", 9),
        anchor="w",
        wraplength=1040,
    ).pack(fill="x", pady=(4, 0))

    table_frame = tk.Frame(result_win, bg="#f4f7f5")
    table_frame.pack(fill="both", expand=True, padx=14, pady=(0, 10))
    columns = ("file", "count", "area", "accepted", "rejected", "raw", "dpi", "note")
    tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
    yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    xscroll = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

    headings = {
        "file": ("File", 300),
        "count": ("Count", 70),
        "area": ("Avg seed mm^2", 110),
        "accepted": ("Accepted", 80),
        "rejected": ("Rejected", 80),
        "raw": ("Raw", 70),
        "dpi": ("DPI", 70),
        "note": ("Note", 320),
    }
    for key, (text, width) in headings.items():
        tree.heading(key, text=text)
        tree.column(key, width=width, minwidth=50, stretch=key in {"file", "note"})

    row_by_iid = {}
    for index, row in enumerate(rows):
        stats = row["stats"]
        iid = str(index)
        row_by_iid[iid] = row
        tree.insert(
            "",
            tk.END,
            iid=iid,
            values=(
                stats.get("fileName", Path(row["path"]).name),
                stats.get("sscount", ""),
                _format_number(stats.get("AvgSizeOfOneSeed", "")),
                stats.get("AcceptedObjectCount", ""),
                stats.get("RejectedObjectCount", ""),
                stats.get("RawObjectCount", ""),
                _format_number(stats.get("CalibrationPPI", ""), 1),
                _short_note(stats.get("ProcessingNote") or stats.get("CalibrationNote")),
            ),
        )

    tree.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll.grid(row=1, column=0, sticky="ew")
    table_frame.columnconfigure(0, weight=1)
    table_frame.rowconfigure(0, weight=1)

    footer = tk.Frame(result_win, bg="#f4f7f5")
    footer.pack(fill="x", padx=14, pady=(0, 12))
    status = tk.StringVar(value=f"{len(rows)} scan result{'s' if len(rows) != 1 else ''}")
    tk.Label(
        footer,
        textvariable=status,
        bg="#f4f7f5",
        fg="#52615c",
        font=("Segoe UI", 9),
        anchor="w",
    ).pack(side="left", fill="x", expand=True)

    def selected_row():
        selection = tree.selection()
        if not selection:
            messagebox.showinfo("SeedSizer Results", "Select a scan row first.", parent=result_win)
            return None
        return row_by_iid[selection[0]]

    def view_overlay():
        row = selected_row()
        if row:
            default_dir = Path(row.get("output_csv")).parent / "overlays" if row.get("output_csv") else None
            _show_overlay_window(result_win, row["path"], calibration=calibration, default_dir=default_dir)

    overlay_button = ttk.Button(footer, text="View Overlay", command=view_overlay)
    overlay_button.pack(side="right")
    tree.bind("<Double-1>", lambda _event: view_overlay())


def _draw_start_logo(canvas):
    canvas.delete("all")
    w = int(canvas.winfo_width() or 620)
    h = int(canvas.winfo_height() or 190)
    cx = w // 2

    canvas.create_rectangle(0, 0, w, h, fill="#101820", outline="")
    canvas.create_oval(cx - 210, -120, cx + 210, 300, fill="#152c35", outline="")
    canvas.create_rectangle(cx - 170, 58, cx + 170, 132, fill="#20343b", outline="#5fd0a5", width=2)
    canvas.create_rectangle(cx - 145, 78, cx + 145, 112, fill="#0d1518", outline="#31494f")
    canvas.create_line(cx - 130, 95, cx + 130, 95, fill="#5fd0a5", width=2)

    seed_colors = ["#caa45a", "#d8b96f", "#b98f45", "#e0c47d", "#c79b52"]
    seed_points = [
        (-92, 20, 13, 8), (-53, -3, 12, 7), (-18, 22, 14, 8),
        (23, -8, 13, 8), (61, 18, 12, 7), (96, -2, 13, 8),
    ]
    for i, (dx, dy, rx, ry) in enumerate(seed_points):
        x, y = cx + dx, 95 + dy
        canvas.create_oval(x - rx, y - ry, x + rx, y + ry, fill=seed_colors[i % len(seed_colors)],
                           outline="#f1d894", width=1)
        canvas.create_line(x - rx // 2, y, x + rx // 2, y, fill="#8f6d35", width=1)

    canvas.create_text(cx, 28, text="SeedSizer", fill="#f4fbf7", font=("Segoe UI", 30, "bold"))
    canvas.create_text(cx, 156, text="Scan. Count. Export.", fill="#b8cbc5", font=("Segoe UI", 12))


def _run_single_folder_gui(root, output_dir=None, calibration=None):
    folder = filedialog.askdirectory(title="Select folder containing .TIFF images")
    if folder:
        summary = Cycle(folder, root, close_progress=False, output_dir=output_dir, calibration=calibration)
        if summary and summary["rows"]:
            _show_results_window(
                root,
                summary["rows"],
                title=f"SeedSizer Results - {Path(folder).name}",
                output_csv=summary["output_csv"],
                calibration=calibration,
            )


def _run_folder_queue_gui(root, default_output_dir=None, calibration=None):
    selected_folders = []
    output_folders = []
    queue_states = []
    is_running = False

    queue_win = tk.Toplevel(root)
    queue_win.title("SeedSizer Folder Queue")
    queue_win.geometry("920x520")
    queue_win.minsize(720, 420)

    label = tk.Label(queue_win, text="Add folders to process in order.", font=("Segoe UI", 11))
    label.pack(anchor="w", padx=12, pady=(12, 6))

    list_frame = tk.Frame(queue_win)
    list_frame.pack(fill="both", expand=True, padx=12, pady=6)

    scrollbar = tk.Scrollbar(list_frame)
    scrollbar.pack(side="right", fill="y")

    listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, yscrollcommand=scrollbar.set)
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=listbox.yview)

    def refresh_list():
        listbox.delete(0, tk.END)
        for index, folder in enumerate(selected_folders):
            state = queue_states[index] if index < len(queue_states) else "Pending"
            output_folder = output_folders[index] if index < len(output_folders) else None
            output_text = str(output_folder) if output_folder else str(_default_output_dir(folder))
            listbox.insert(tk.END, f"{state:<8} {folder}  ->  {output_text}")
            if state == "Running":
                listbox.itemconfig(index, foreground="#005a9e")
            elif state == "Done":
                listbox.itemconfig(index, foreground="#4f6f52")
            elif state in {"Skipped", "Error"}:
                listbox.itemconfig(index, foreground="#a33a3a")

    def add_folder():
        if is_running:
            return
        folder = filedialog.askdirectory(title="Add folder containing .TIFF images")
        if folder and Path(folder) not in selected_folders:
            folder_path = Path(folder)
            selected_folders.append(folder_path)
            output_folders.append(Path(default_output_dir) if default_output_dir else None)
            queue_states.append("Pending")
            refresh_list()

    def remove_selected():
        if is_running:
            return
        for index in reversed(listbox.curselection()):
            del selected_folders[index]
            del output_folders[index]
            del queue_states[index]
        refresh_list()

    def clear_queue():
        if is_running:
            return
        selected_folders.clear()
        output_folders.clear()
        queue_states.clear()
        refresh_list()

    def set_selected_output_folder():
        if is_running:
            return
        selected = list(listbox.curselection())
        if not selected:
            return
        folder = filedialog.askdirectory(title="Select output folder for selected queue item(s)")
        if not folder:
            return
        for index in selected:
            output_folders[index] = Path(folder)
        refresh_list()

    def reset_selected_output_folder():
        if is_running:
            return
        selected = list(listbox.curselection())
        if not selected:
            return
        for index in selected:
            output_folders[index] = None
        refresh_list()

    status_frame = tk.Frame(queue_win)
    status_frame.pack(fill="x", padx=12, pady=(2, 6))

    label_status = tk.Label(status_frame, text="Queue idle", anchor="w", font=("Segoe UI", 10))
    label_status.pack(fill="x")

    label_file = tk.Label(status_frame, text="", anchor="w", font=("Segoe UI", 9))
    label_file.pack(fill="x", pady=(2, 0))

    progress_bar = ttk.Progressbar(status_frame, length=500, mode="determinate")
    progress_bar.pack(fill="x", pady=(8, 0))

    button_frame = tk.Frame(queue_win)
    button_frame.pack(fill="x", padx=12, pady=(6, 12))

    button_add = tk.Button(button_frame, text="Add Folder", command=add_folder)
    button_remove = tk.Button(button_frame, text="Remove Selected", command=remove_selected)
    button_clear = tk.Button(button_frame, text="Clear", command=clear_queue)
    button_output = tk.Button(button_frame, text="Set Output", command=set_selected_output_folder)
    button_default_output = tk.Button(button_frame, text="Default Output", command=reset_selected_output_folder)
    button_start = tk.Button(button_frame, text="Start Queue")

    button_add.pack(side="left")
    button_remove.pack(side="left", padx=(8, 0))
    button_clear.pack(side="left", padx=(8, 0))
    button_output.pack(side="left", padx=(8, 0))
    button_default_output.pack(side="left", padx=(8, 0))
    button_start.pack(side="right", padx=(0, 8))

    def set_queue_controls(enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        for button in (
            button_add, button_remove, button_clear, button_output,
            button_default_output, button_start,
        ):
            button.config(state=state)

    def start_queue():
        nonlocal is_running
        if selected_folders:
            is_running = True
            set_queue_controls(False)
            queue_win.protocol("WM_DELETE_WINDOW", lambda: None)
            total_folders = len(selected_folders)
            reserved_outputs = set()
            queue_results = []

            for folder_index, folder in enumerate(list(selected_folders)):
                queue_states[folder_index] = "Running"
                refresh_list()
                listbox.selection_clear(0, tk.END)
                listbox.selection_set(folder_index)
                listbox.see(folder_index)
                final_state = {"value": "Done"}

                def update_progress(event, **info):
                    total_files = info.get("total_files", 0)
                    file_index = info.get("file_index", 0)
                    output_csv = info.get("output_csv")
                    file_name = info.get("file_name", "")
                    progress_bar["maximum"] = max(total_files, 1)
                    progress_bar["value"] = file_index
                    if event == "folder_start":
                        label_status.config(text=f"Folder {folder_index + 1}/{total_folders}: {folder.name}")
                        label_file.config(text=f"{total_files} TIFF files found")
                    elif event == "file_start":
                        label_status.config(text=f"Folder {folder_index + 1}/{total_folders}: {folder.name}")
                        label_file.config(text=f"Processing {file_index + 1}/{total_files}: {file_name}")
                    elif event == "file_done":
                        label_file.config(text=f"Completed {file_index}/{total_files}: {file_name}")
                    elif event == "folder_done":
                        label_status.config(text=f"Saved {output_csv.name}")
                        label_file.config(text="")
                    elif event in {"folder_empty", "folder_error"}:
                        final_state["value"] = "Skipped" if event == "folder_empty" else "Error"
                        label_status.config(text=info.get("message", "Folder skipped"))
                        label_file.config(text=str(info.get("folder_path", "")))
                    queue_win.update()

                summary = Cycle(
                    folder,
                    progress_callback=update_progress,
                    output_dir=output_folders[folder_index],
                    reserved_outputs=reserved_outputs,
                    calibration=calibration,
                )
                if summary:
                    queue_results.extend(summary["rows"])
                queue_states[folder_index] = final_state["value"]
                refresh_list()

            progress_bar["value"] = progress_bar["maximum"]
            label_status.config(text="Queue complete")
            label_file.config(text="")
            is_running = False
            queue_win.protocol("WM_DELETE_WINDOW", queue_win.destroy)
            set_queue_controls(True)
            if queue_results:
                _show_results_window(
                    queue_win,
                    queue_results,
                    title="SeedSizer Queue Results",
                    calibration=calibration,
                )

    def cancel_queue():
        if not is_running:
            queue_win.destroy()

    button_start.config(command=start_queue)

    queue_win.protocol("WM_DELETE_WINDOW", cancel_queue)
    queue_win.grab_set()
    queue_win.wait_window()


def _synthetic_seed_gray(brightness):
    brightness = float(np.clip(brightness, 0, 100))
    lo = 42.0 + 0.48 * brightness
    hi = min(155.0, lo + 18.0 + 0.22 * brightness)
    return lo, hi


def _format_number(value, digits=3):
    if value == "" or pd.isna(value):
        return "unavailable"
    return f"{float(value):.{digits}f}"


def _run_test_seedsizer_gui(root, calibration=None):
    try:
        import synth_test
    except Exception as exc:
        messagebox.showerror("Test SeedSizer", f"Could not load the synthetic data producer:\n{exc}")
        return

    test_win = tk.Toplevel(root)
    test_win.title("Test SeedSizer")
    test_win.geometry("1120x760")
    test_win.minsize(960, 660)
    test_win.configure(bg="#f4f7f5")

    style = ttk.Style(test_win)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("TestPrimary.TButton", font=("Segoe UI", 11), padding=(14, 9))

    shell = tk.Frame(test_win, bg="#f4f7f5")
    shell.pack(fill="both", expand=True, padx=18, pady=16)
    shell.columnconfigure(1, weight=1)
    shell.rowconfigure(1, weight=1)

    tk.Label(
        shell,
        text="Test SeedSizer With Synthetic Seeds",
        bg="#f4f7f5",
        fg="#24322e",
        font=("Segoe UI", 16, "bold"),
    ).grid(row=0, column=0, columnspan=2, sticky="w")

    controls = tk.Frame(shell, bg="#f4f7f5", width=330)
    controls.grid(row=1, column=0, sticky="nsw", pady=(14, 0), padx=(0, 16))
    controls.columnconfigure(1, weight=1)

    output = tk.Frame(shell, bg="#f4f7f5")
    output.grid(row=1, column=1, sticky="nsew", pady=(14, 0))
    output.columnconfigure(0, weight=1)
    output.rowconfigure(0, weight=1)

    seed_count = tk.IntVar(value=120)
    clumped_pct = tk.DoubleVar(value=35)
    cluster_size = tk.IntVar(value=8)
    seed_scale = tk.DoubleVar(value=1.0)
    brightness = tk.DoubleVar(value=70)
    dust_count = tk.IntVar(value=120)
    banding = tk.DoubleVar(value=2.5)
    random_seed = tk.IntVar(value=7)
    bed_pixels = tk.IntVar(value=2800)
    status = tk.StringVar(value="Ready")
    last = {
        "photo": None,
        "scan_path": None,
        "raw_image": None,
        "overlay_image": None,
        "overlay_visible": False,
    }
    busy_widgets = []

    def add_row(row, label, widget):
        tk.Label(
            controls,
            text=label,
            bg="#f4f7f5",
            fg="#2c3d37",
            font=("Segoe UI", 10),
        ).grid(row=row, column=0, sticky="w", pady=5)
        widget.grid(row=row, column=1, sticky="ew", pady=5)
        busy_widgets.append(widget)

    row = 0
    add_row(row, "Number of seeds", tk.Spinbox(
        controls, from_=1, to=2500, increment=1, textvariable=seed_count, width=9))

    row += 1
    add_row(row, "Clumped seeds (%)", ttk.Scale(
        controls, from_=0, to=100, variable=clumped_pct, orient="horizontal"))
    clumped_label = tk.Label(controls, text="", bg="#f4f7f5", fg="#52615c", width=5)
    clumped_label.grid(row=row, column=2, sticky="e", padx=(6, 0))

    row += 1
    add_row(row, "Max cluster size", tk.Spinbox(
        controls, from_=2, to=40, increment=1, textvariable=cluster_size, width=9))

    row += 1
    add_row(row, "Seed size scale", tk.Spinbox(
        controls, from_=0.35, to=1.8, increment=0.05, textvariable=seed_scale, width=9))

    row += 1
    add_row(row, "Seed visibility", ttk.Scale(
        controls, from_=0, to=100, variable=brightness, orient="horizontal"))
    brightness_label = tk.Label(controls, text="", bg="#f4f7f5", fg="#52615c", width=8)
    brightness_label.grid(row=row, column=2, sticky="e", padx=(6, 0))

    row += 1
    add_row(row, "Dust/chaff objects", tk.Spinbox(
        controls, from_=0, to=5000, increment=10, textvariable=dust_count, width=9))

    row += 1
    add_row(row, "Scanner banding", tk.Spinbox(
        controls, from_=0.0, to=8.0, increment=0.25, textvariable=banding, width=9))

    row += 1
    add_row(row, "Random seed", tk.Spinbox(
        controls, from_=0, to=999999, increment=1, textvariable=random_seed, width=9))

    row += 1
    add_row(row, "Canvas pixels", tk.Spinbox(
        controls, from_=1200, to=6000, increment=200, textvariable=bed_pixels, width=9))

    row += 1
    button_box = tk.Frame(controls, bg="#f4f7f5")
    button_box.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(16, 6))
    button_box.columnconfigure(0, weight=1)
    button_box.columnconfigure(1, weight=1)

    preview_button = ttk.Button(button_box, text="Generate Preview", style="TestPrimary.TButton")
    run_button = ttk.Button(button_box, text="Run SeedSizer Test", style="TestPrimary.TButton")
    overlay_button = ttk.Button(button_box, text="Show Detection Overlay", state=tk.DISABLED)
    preview_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
    run_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))
    overlay_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
    busy_widgets.extend([preview_button, run_button])

    row += 1
    tk.Label(
        controls,
        textvariable=status,
        bg="#f4f7f5",
        fg="#52615c",
        font=("Segoe UI", 9),
        anchor="w",
        justify="left",
        wraplength=320,
    ).grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 8))

    row += 1
    results = tk.Text(
        controls,
        height=18,
        width=42,
        wrap="word",
        state="disabled",
        bg="#ffffff",
        fg="#23302c",
        relief="solid",
        borderwidth=1,
        font=("Segoe UI", 9),
    )
    results.grid(row=row, column=0, columnspan=3, sticky="nsew")
    controls.rowconfigure(row, weight=1)

    preview_frame = tk.Frame(output, bg="#101820", highlightthickness=1, highlightbackground="#20343b")
    preview_frame.grid(row=0, column=0, sticky="nsew")
    preview_frame.columnconfigure(0, weight=1)
    preview_frame.rowconfigure(0, weight=1)

    preview_label = tk.Label(
        preview_frame,
        text="Generate a preview to see synthetic seeds here.",
        bg="#101820",
        fg="#c7d8d2",
        font=("Segoe UI", 12),
        anchor="center",
    )
    preview_label.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

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

    def update_slider_labels(*_):
        clumped_label.config(text=f"{int(clumped_pct.get())}%")
        lo, hi = _synthetic_seed_gray(brightness.get())
        brightness_label.config(text=f"{int(lo)}-{int(hi)}")

    clumped_pct.trace_add("write", update_slider_labels)
    brightness.trace_add("write", update_slider_labels)
    update_slider_labels()

    def selected_params():
        n = max(1, int(seed_count.get()))
        frac = float(clumped_pct.get()) / 100.0
        max_cluster = max(2, int(cluster_size.get()))
        csize = (2, max_cluster) if frac > 0 else (0, 0)
        scale = max(0.05, float(seed_scale.get()))
        gray = _synthetic_seed_gray(brightness.get())
        return (
            n,
            frac,
            csize,
            float(banding.get()),
            max(0, int(dust_count.get())),
            scale,
            gray,
        )

    def display_image(img):
        pil = img.copy() if isinstance(img, Image.Image) else Image.fromarray(img)
        pil.thumbnail((740, 690), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        preview_label.config(image=photo, text="")
        preview_label.image = photo
        last["photo"] = photo

    def show_raw_preview():
        if last["raw_image"] is not None:
            display_image(last["raw_image"])
            last["overlay_visible"] = False
            overlay_button.config(text="Show Detection Overlay")

    def show_detection_overlay():
        if last["overlay_image"] is not None:
            display_image(last["overlay_image"])
            last["overlay_visible"] = True
            overlay_button.config(text="Hide Detection Overlay")

    def toggle_detection_overlay():
        if last["overlay_visible"]:
            show_raw_preview()
        else:
            show_detection_overlay()

    def output_path():
        out_dir = Path(__file__).resolve().parent / "SynthScans"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return out_dir / f"seedsizer_test_{stamp}_{int(seed_count.get())}seeds.png"

    def format_results(truth, measured=None, path=None):
        requested = int(seed_count.get())
        placed = int(truth["n_seeds"])
        touching = int(truth["n_touching"])
        overlap = float(truth["overlap_pct"])
        missed_placement = requested - placed
        lines = [
            "Synthetic truth",
            f"Requested seeds: {requested}",
            f"Placed seeds: {placed}",
            f"Clumped/touching seeds: {touching} ({int(clumped_pct.get())}% requested)",
            f"Seed footprint overlap: {overlap:+.2f}% ({'pass' if abs(overlap) < 5 else 'inspect'})",
            f"Truth mean area: {truth['mean_area_mm2']:.3f} mm^2",
            f"Dust/chaff objects: {truth['n_dust']}",
            f"Canvas: {truth['bed']}",
        ]
        if placed != requested:
            lines.append(
                f"Placement note: {missed_placement} requested seeds were not drawn. "
                "Increase canvas pixels, lower seed count, lower seed size, or reduce clumping."
            )
        if measured is not None:
            got = int(measured["sscount"])
            placed_err = 100.0 * (got - placed) / max(1, placed)
            lines.extend([
                "",
                "SeedSizer output",
                f"Detected count: {got}",
                f"Difference vs placed truth: {got - placed:+d} ({placed_err:+.2f}%)",
                f"Average single-seed area: {_format_number(measured['AvgSizeOfOneSeed'])} mm^2",
                f"Accepted objects: {measured['AcceptedObjectCount']}",
                f"Rejected objects: {measured['RejectedObjectCount']}",
                f"Raw threshold objects: {measured['RawObjectCount']}",
                f"Calibration DPI: {_format_number(measured['CalibrationPPI'], 1)}",
            ])
            if measured["CalibrationNote"]:
                lines.append(f"Calibration note: {measured['CalibrationNote']}")
            if measured["ProcessingNote"]:
                lines.append(f"Processing note: {measured['ProcessingNote']}")
        if path is not None:
            lines.extend(["", f"Generated file: {path}"])
        return "\n".join(lines)

    def run_test(analyze):
        try:
            params = selected_params()
            px = max(1200, int(bed_pixels.get()))
            seed = int(random_seed.get())
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        set_busy(True)
        overlay_button.config(state=tk.DISABLED, text="Show Detection Overlay")
        last["scan_path"] = None
        last["overlay_image"] = None
        last["overlay_visible"] = False
        status.set("Generating synthetic scan...")

        def worker():
            try:
                rng = np.random.default_rng(seed)
                img, truth = synth_test.generate("seedsizer_test", rng, (px, px), params)
                measured = None
                saved_path = None
                overlay = None
                if analyze:
                    saved_path = output_path()
                    Image.fromarray(img).save(saved_path, dpi=(PPI, PPI))
                    test_win.after(0, lambda: status.set("Running SeedSizer on the generated scan..."))
                    test_win.after(0, lambda: status.set("Building detection overlay..."))
                    with contextlib.redirect_stdout(io.StringIO()):
                        overlay, measured = CreateOverlay(saved_path, calibration=calibration)
                test_win.after(0, lambda: done(img, truth, measured, saved_path, overlay, None))
            except Exception as exc:
                test_win.after(0, lambda exc=exc: done(None, None, None, None, None, exc))

        def done(img, truth, measured, saved_path, overlay, err):
            set_busy(False)
            if err is not None:
                status.set("Test failed")
                messagebox.showerror("Test SeedSizer", str(err))
                return
            last["raw_image"] = img
            last["overlay_image"] = overlay
            last["overlay_visible"] = False
            if overlay is not None:
                show_detection_overlay()
            else:
                display_image(img)
            set_results(format_results(truth, measured, saved_path))
            last["scan_path"] = saved_path if measured is not None else None
            overlay_button.config(state=tk.NORMAL if overlay is not None else tk.DISABLED)
            status.set("SeedSizer test complete" if measured is not None else "Preview generated")

        threading.Thread(target=worker, daemon=True).start()

    preview_button.config(command=lambda: run_test(False))
    run_button.config(command=lambda: run_test(True))
    overlay_button.config(command=toggle_detection_overlay)

    set_results("No synthetic scan has been generated yet.")
    test_win.grab_set()


def _run_start_menu_gui(root):
    root.title("SeedSizer")
    root.geometry("720x800")
    root.minsize(620, 760)
    root.configure(bg="#f4f7f5")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("Start.TButton", font=("Segoe UI", 12), padding=(16, 12))
    style.configure("Exit.TButton", font=("Segoe UI", 11), padding=(14, 10))
    output_dir = {"path": None}
    output_text = tk.StringVar(value="Output folder: default next to each scan folder")
    calibration_mode = tk.StringVar(value="Automatic")
    calibration_dpi = tk.DoubleVar(value=PPI)
    calibration_text = tk.StringVar()

    shell = tk.Frame(root, bg="#f4f7f5")
    shell.pack(fill="both", expand=True, padx=24, pady=22)
    shell.columnconfigure(0, weight=1)
    shell.rowconfigure(1, weight=1)

    logo = tk.Canvas(shell, height=190, highlightthickness=0, bg="#101820")
    logo.grid(row=0, column=0, sticky="ew")
    logo.bind("<Configure>", lambda _event: _draw_start_logo(logo))

    panel = tk.Frame(shell, bg="#f4f7f5")
    panel.grid(row=1, column=0, sticky="nsew", pady=(20, 0))
    panel.columnconfigure(0, weight=1)

    tk.Label(
        panel,
        text="Choose how you want to process seed scans.",
        bg="#f4f7f5",
        fg="#26332f",
        font=("Segoe UI", 14, "bold"),
    ).grid(row=0, column=0, sticky="w")

    tk.Label(
        panel,
        text="Single-folder analysis writes one CSV for one scan folder. Multi-folder queue batches several folders. Test SeedSizer generates known synthetic seeds and compares truth to detection.",
        bg="#f4f7f5",
        fg="#52615c",
        font=("Segoe UI", 10),
        wraplength=620,
        justify="left",
    ).grid(row=1, column=0, sticky="w", pady=(6, 16))

    output_panel = tk.Frame(panel, bg="#e8efeb", highlightthickness=1, highlightbackground="#c9d6d0")
    output_panel.grid(row=2, column=0, sticky="ew", pady=(0, 16))
    output_panel.columnconfigure(0, weight=1)

    tk.Label(
        output_panel,
        textvariable=output_text,
        bg="#e8efeb",
        fg="#2c3d37",
        font=("Segoe UI", 9),
        anchor="w",
        wraplength=500,
        justify="left",
    ).grid(row=0, column=0, sticky="ew", padx=10, pady=9)

    def choose_output_folder():
        folder = filedialog.askdirectory(title="Select output folder for CSV files")
        if folder:
            output_dir["path"] = Path(folder)
            output_text.set(f"Output folder: {output_dir['path']}")

    def use_default_output_folder():
        output_dir["path"] = None
        output_text.set("Output folder: default next to each scan folder")

    output_buttons = tk.Frame(output_panel, bg="#e8efeb")
    output_buttons.grid(row=0, column=1, sticky="e", padx=10, pady=8)
    ttk.Button(output_buttons, text="Choose", command=choose_output_folder).pack(side="left")
    ttk.Button(output_buttons, text="Default", command=use_default_output_folder).pack(side="left", padx=(8, 0))

    calibration_panel = tk.Frame(panel, bg="#e8efeb", highlightthickness=1, highlightbackground="#c9d6d0")
    calibration_panel.grid(row=3, column=0, sticky="ew", pady=(0, 16))
    calibration_panel.columnconfigure(1, weight=1)

    tk.Label(
        calibration_panel,
        text="Calibration",
        bg="#e8efeb",
        fg="#2c3d37",
        font=("Segoe UI", 10, "bold"),
    ).grid(row=0, column=0, sticky="w", padx=10, pady=(9, 4))

    mode_box = ttk.Combobox(
        calibration_panel,
        textvariable=calibration_mode,
        values=("Automatic", "Manual DPI", "Use 1200 DPI"),
        state="readonly",
        width=14,
    )
    mode_box.grid(row=0, column=1, sticky="w", padx=(0, 10), pady=(9, 4))

    tk.Label(
        calibration_panel,
        text="DPI",
        bg="#e8efeb",
        fg="#2c3d37",
        font=("Segoe UI", 9),
    ).grid(row=0, column=2, sticky="e", padx=(0, 6), pady=(9, 4))

    dpi_box = tk.Spinbox(
        calibration_panel,
        from_=MIN_PLAUSIBLE_DPI,
        to=MAX_PLAUSIBLE_DPI,
        increment=50,
        textvariable=calibration_dpi,
        width=7,
    )
    dpi_box.grid(row=0, column=3, sticky="e", padx=(0, 10), pady=(9, 4))

    tk.Label(
        calibration_panel,
        textvariable=calibration_text,
        bg="#e8efeb",
        fg="#52615c",
        font=("Segoe UI", 9),
        anchor="w",
        wraplength=620,
        justify="left",
    ).grid(row=1, column=0, columnspan=4, sticky="ew", padx=10, pady=(0, 9))

    def update_calibration_text(*_):
        mode = calibration_mode.get()
        try:
            dpi_value = float(calibration_dpi.get())
        except (tk.TclError, ValueError):
            dpi_value = PPI
        if mode == "Automatic":
            calibration_text.set(
                f"Reads image DPI when reliable; otherwise uses {_format_dpi(dpi_value)} DPI and writes a warning."
            )
        elif mode == "Manual DPI":
            calibration_text.set(f"Always measures area and length using {_format_dpi(dpi_value)} DPI.")
        else:
            calibration_text.set(f"Always measures area and length using the SeedSizer default {_format_dpi(PPI)} DPI.")

    def current_calibration():
        mode_map = {
            "Automatic": "auto",
            "Manual DPI": "manual",
            "Use 1200 DPI": "fixed",
        }
        try:
            dpi_value = float(calibration_dpi.get())
        except (tk.TclError, ValueError):
            dpi_value = PPI
        return {
            "mode": mode_map.get(calibration_mode.get(), "auto"),
            "default_dpi": dpi_value if calibration_mode.get() == "Automatic" else PPI,
            "manual_dpi": dpi_value,
        }

    calibration_mode.trace_add("write", update_calibration_text)
    calibration_dpi.trace_add("write", update_calibration_text)
    update_calibration_text()

    buttons = tk.Frame(panel, bg="#f4f7f5")
    buttons.grid(row=4, column=0, sticky="ew")
    buttons.columnconfigure(0, weight=1)

    single_button = ttk.Button(
        buttons,
        text="Start Scanning",
        style="Start.TButton",
        command=lambda: _run_single_folder_gui(root, output_dir["path"], current_calibration()),
    )
    queue_button = ttk.Button(
        buttons,
        text="Multi-Folder Queue",
        style="Start.TButton",
        command=lambda: _run_folder_queue_gui(root, output_dir["path"], current_calibration()),
    )
    test_button = ttk.Button(
        buttons,
        text="Test SeedSizer",
        style="Start.TButton",
        command=lambda: _run_test_seedsizer_gui(root, current_calibration()),
    )
    exit_button = ttk.Button(
        buttons,
        text="Exit",
        style="Exit.TButton",
        command=root.destroy,
    )
    single_button.grid(row=0, column=0, sticky="ew", pady=(0, 10))
    queue_button.grid(row=1, column=0, sticky="ew", pady=(0, 10))
    test_button.grid(row=2, column=0, sticky="ew", pady=(0, 10))
    exit_button.grid(row=3, column=0, sticky="ew", pady=(10, 0))

    tk.Label(
        panel,
        text="Default CSV names stay <folder>_data.csv. Custom output folders add a short folder tag to avoid collisions.",
        bg="#f4f7f5",
        fg="#6c7a75",
        font=("Segoe UI", 9),
    ).grid(row=5, column=0, sticky="w", pady=(16, 0))

    root.mainloop()


# Run SeedSizer.py as a standalone program
# You can also convert to a .exe using pyinstaller and run on any device without Python installed

# Note folder query may take a few seconds due to filedialog performance on some systems

if __name__ == "__main__":
    root = None
    if len(sys.argv) > 1:
        folders = sys.argv[1:]
    else:
        try:
            root = tk.Tk()
            _run_start_menu_gui(root)
            sys.exit(0)
        except tk.TclError as exc:
            print(f"Could not open SeedSizer start menu ({exc}).")
            print("Run with folder paths instead, for example: python3 SeedSizer.py /path/to/scans /path/to/more/scans")
            sys.exit(1)

    if not folders:
        print("No folders selected. Exiting.")
        sys.exit(0)

    CycleQueue(folders, root)
