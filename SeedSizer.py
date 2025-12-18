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
Image.MAX_IMAGE_PIXELS = None  # disable Pillow’s decompression bomb limit
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


def Run(filename):

    ### Image Manipulation ###

    path = Path(filename).resolve()
    raw_image = iio.imread(path)                                                        
    grayscale_image = np.mean(raw_image, axis=2).astype(np.float32)                     # Loads as float32 Grayscale image, which is a decimal number of 0 to 1 for how dark the pixel is
                                                                                        # The seed's are closer to white and the background is closer to black, so we want to find the ideal threshold.
    del raw_image
    gc.collect() 

    threshold_value = threshold_otsu(grayscale_image)                                   # Applies Otsu's thresholding, just finds best threshold between seeds and background.
    binary_image = grayscale_image > threshold_value                                    # This labels each pixel in the background as either 1 (True) or 0 (False), where 1 is the seed and 0 is the background
    binary_clean = remove_small_objects(binary_image, min_size = int(PP_SQMM*FILTER))   # Reduces noise by removing small artifacts
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

    df["area_mm2"] = df["area"] / PP_SQMM                                               # Convert these connected components to mm^2 since we know pixel is 1/1200 of an inch
    df["aspect_ratio"] = df["major_axis_length"] / df["minor_axis_length"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    mean_alpha = df ["area_mm2"].mean()                                                 # Finding average size of connected components
    median_area = df["area_mm2"].median()

    ### Statistical Analysis ###

    df_filtered = df[df["area_mm2"] >= MINSIZE * median_area]                           # Selecting objets larger than 40% of the median area
    clumps = df_filtered[df_filtered["area_mm2"] > 1.9 * mean_alpha].copy()             # Selecting objects larger than 1.9 times the mean area (these are likely clumps of seeds)
    clumps["clump_size"] = (clumps["area_mm2"] / mean_alpha).round().astype(int)        # Estimating number of seeds in each clump

    size_clumps = clumps["clump_size"].sum()                                            # Counting number of seeds in clumps
    size_singles = len(df_filtered[df_filtered["area_mm2"] <= 1.9 * mean_alpha])        # Counting number of single seeds (not in clumps)
    total_size = size_clumps + size_singles                                             # Aggregate seed count                                      
    total_area = df_filtered["area_mm2"].sum()

    mean_beta = df_filtered[df_filtered["area_mm2"] <= 1.9 * mean_alpha]["area_mm2"].mean() 

    length_mm = df_filtered["major_axis_length"] / PX_PER_MM
    width_mm  = df_filtered["minor_axis_length"] / PX_PER_MM

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
    print(f"Average seed size: {mean_beta:.3f} mm²")
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
        "AvgSizeOfOneSeed": int(mean_beta),
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
        progress_bar["value"] = i - 1
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