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

from PIL import Image, ImageTk
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
    # No seed layer can cover 15% of the bed, so treat that as the tell and use
    # a fixed threshold that keeps dim seed bodies intact instead of isolating
    # only bright cores (Pot_163: ~16 real seeds reported as 32 at threshold 62).
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
          output_dir=None, reserved_outputs=None):
    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"Selected folder does not exist: {folder_path}")
        if progress_callback:
            progress_callback("folder_error", folder_path=folder_path, message="Folder does not exist")
        return

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
        return

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

        stats = Run(tif_file)

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


def CycleQueue(folders, root=None, output_dirs=None):
    folders = [Path(folder) for folder in folders if folder]
    output_dirs = list(output_dirs or [None] * len(folders))
    reserved_outputs = set()
    total_folders = len(folders)
    for index, folder in enumerate(folders, start=1):
        print(f"Starting folder {index}/{total_folders}: {folder}", flush=True)
        output_dir = output_dirs[index - 1] if index - 1 < len(output_dirs) else None
        Cycle(folder, root, close_progress=index < total_folders,
              output_dir=output_dir, reserved_outputs=reserved_outputs)


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


def _run_single_folder_gui(root, output_dir=None):
    folder = filedialog.askdirectory(title="Select folder containing .TIFF images")
    if folder:
        Cycle(folder, root, close_progress=False, output_dir=output_dir)


def _run_folder_queue_gui(root, default_output_dir=None):
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

                Cycle(
                    folder,
                    progress_callback=update_progress,
                    output_dir=output_folders[folder_index],
                    reserved_outputs=reserved_outputs,
                )
                queue_states[folder_index] = final_state["value"]
                refresh_list()

            progress_bar["value"] = progress_bar["maximum"]
            label_status.config(text="Queue complete")
            label_file.config(text="")
            is_running = False
            queue_win.protocol("WM_DELETE_WINDOW", queue_win.destroy)

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


def _run_test_seedsizer_gui(root):
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
    last = {"photo": None}
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
    preview_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
    run_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))
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
        pil = Image.fromarray(img)
        pil.thumbnail((740, 690), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        preview_label.config(image=photo, text="")
        preview_label.image = photo
        last["photo"] = photo

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
            requested_err = 100.0 * (got - requested) / max(1, requested)
            lines.extend([
                "",
                "SeedSizer output",
                f"Detected count: {got}",
                f"Difference vs placed truth: {got - placed:+d} ({placed_err:+.2f}%)",
                f"Difference vs requested setting: {got - requested:+d} ({requested_err:+.2f}%)",
                f"Average single-seed area: {_format_number(measured['AvgSizeOfOneSeed'])} mm^2",
                f"Accepted objects: {measured['AcceptedObjectCount']}",
                f"Rejected objects: {measured['RejectedObjectCount']}",
                f"Raw threshold objects: {measured['RawObjectCount']}",
            ])
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
        status.set("Generating synthetic scan...")

        def worker():
            try:
                rng = np.random.default_rng(seed)
                img, truth = synth_test.generate("seedsizer_test", rng, (px, px), params)
                measured = None
                saved_path = None
                if analyze:
                    saved_path = output_path()
                    Image.fromarray(img).save(saved_path)
                    test_win.after(0, lambda: status.set("Running SeedSizer on the generated scan..."))
                    with contextlib.redirect_stdout(io.StringIO()):
                        measured = Run(saved_path)
                test_win.after(0, lambda: done(img, truth, measured, saved_path, None))
            except Exception as exc:
                test_win.after(0, lambda exc=exc: done(None, None, None, None, exc))

        def done(img, truth, measured, saved_path, err):
            set_busy(False)
            if err is not None:
                status.set("Test failed")
                messagebox.showerror("Test SeedSizer", str(err))
                return
            display_image(img)
            set_results(format_results(truth, measured, saved_path))
            status.set("SeedSizer test complete" if measured is not None else "Preview generated")

        threading.Thread(target=worker, daemon=True).start()

    preview_button.config(command=lambda: run_test(False))
    run_button.config(command=lambda: run_test(True))

    set_results("No synthetic scan has been generated yet.")
    test_win.grab_set()


def _run_start_menu_gui(root):
    root.title("SeedSizer")
    root.geometry("720x710")
    root.minsize(620, 680)
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

    buttons = tk.Frame(panel, bg="#f4f7f5")
    buttons.grid(row=3, column=0, sticky="ew")
    buttons.columnconfigure(0, weight=1)

    single_button = ttk.Button(
        buttons,
        text="Start Scanning",
        style="Start.TButton",
        command=lambda: _run_single_folder_gui(root, output_dir["path"]),
    )
    queue_button = ttk.Button(
        buttons,
        text="Multi-Folder Queue",
        style="Start.TButton",
        command=lambda: _run_folder_queue_gui(root, output_dir["path"]),
    )
    test_button = ttk.Button(
        buttons,
        text="Test SeedSizer",
        style="Start.TButton",
        command=lambda: _run_test_seedsizer_gui(root),
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
    ).grid(row=4, column=0, sticky="w", pady=(16, 0))

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
