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

PPI = 1200                                                                      # Pixels per inch
PP_SQMM = (PPI / 25.4) ** 2                                                     # Pixels per square millimeter
FILTER = 0.1                                                                    # This is basically used saying we won't accept anything < .1 mm^2 in size
MINSIZE = .5                                                                    # This is the minimum size of a seed we want to consider, in this case 50% of the median area   

#################################################################################################################################################################################################
# I see you're digging around, if you want help contact for assistance: jefferson.kline@wsu.edu
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

    binary_seed = regionprops_table(labeled_image, properties=["label", "area"])        # Collection of area's of the connected components (collection of adjascent pixels labeled 1, which make up the seed)
    df = pd.DataFrame(binary_seed)                                                      # This turns our dictionary of connected components into a pandas dataframe

    df["area_mm2"] = df["area"] / PP_SQMM                                               # Convert these connected components to mm^2 since we know pixel is 1/1200 of an inch
    mean_alpha = df ["area_mm2"].mean()                                                 # Finding average size of connected components
    median_area = df["area_mm2"].median()

    ### Statistical Analysis ###

    df_filtered = df[df["area_mm2"] >= MINSIZE * median_area]                           # Selecting objets larger than 40% of the median area
    clumps = df_filtered[df_filtered["area_mm2"] > 1.9 * mean_alpha].copy()
    clumps["clump_size"] = (clumps["area_mm2"] / mean_alpha).round().astype(int)

    size_clumps = clumps["clump_size"].sum()
    size_singles = len(df_filtered[df_filtered["area_mm2"] <= 1.9 * mean_alpha])
    total_size = size_clumps + size_singles
    total_area = df_filtered["area_mm2"].sum()

    mean_beta = df_filtered[df_filtered["area_mm2"] <= 1.9 * mean_alpha]["area_mm2"].mean()

    ### Output for User ##
    
    print(f"Filepath: {path}")
    print(f"Total number of filtered seeds: {total_size}")                              
    print(f"Average seed size is: {mean_beta}")
    print()

    return total_size, mean_beta, median_area, total_size


def Cycle(folder = "Data"):

    folder_path = Path(folder)
    tif_files = [file for file in folder_path.glob("*.tif")]
    output_csv = "output.csv"

    results = []

    for tif_file in tif_files:
        total_size, mean_beta, median_area, total_size = Run(tif_file.name)

        results.append({
            "Total Size": total_size,
            "Mean Size": mean_beta,
            "Median Area": median_area,
            "File": tif_file.name
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    Cycle(folder)
