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
import os
import gc

PPI = 1200                                                                      # Pixels per inch
PP_SQMM = (PPI / 25.4) ** 2                                                     # Pixels per square millimeter
FILTER = 0.1                                                                    # This is basically used saying we won't accept anything < .1 mm^2 in size
MINSIZE = .5

#################################################################################################################################################################################################

def Run(filename):


    ### Image Manipulation ###

    path = f"Data/{filename}"
    raw_image = tifffile.memmap(path)                                             # Load File, base case here is Data/Calena-18.tif
    grayscale_image = np.mean(raw_image, axis=2).astype(np.float32)                 # Loads as float32 Grayscale image, which is a decimal number of 0 to 1 for how dark the pixel is
                                                                                    # The seed's are closer to white and the background is closer to black, so we want to find the ideal threshold.
    del raw_image
    gc.collect() 

    threshold_value = threshold_otsu(grayscale_image)                                   # Applies Otsu's thresholding, just finds threshold between seeds and background.
    binary_image = grayscale_image > threshold_value                                    # This labels each pixel in the background as either 1 (True) or 0 (False), where 1 is the seed and 0 is the background
    binary_clean = remove_small_objects(binary_image, min_size = int(PP_SQMM*FILTER))   # Reduces noise by removing small artifacts
    labeled_image = label(binary_clean)                                                 


    ### Image Analysis ###

    binary_seed = regionprops_table(labeled_image, properties=["label", "area"])    # Collection of area's of the connected components (collection of adjascent pixels labeled 1, which make up the seed)
    df = pd.DataFrame(binary_seed)                                                  # This turns our dictionary of connected components into a pandas dataframe, where pandas has functions to make this easy
    df["area_mm2"] = df["area"] / PP_SQMM                                           # Convert these connected components to mm^2 since we know pixel is 1/1200 of an inch
    median_area = df["area_mm2"].median()

    df_filtered = df[df["area_mm2"] >= MINSIZE * median_area]                        # Selecting objets smaller than 40% of the median area
    valid_labels = set(df_filtered["label"])
    final_mask = np.isin(labeled_image, list(valid_labels))
    

    ### Watershed segmentation ###

    distance = ndi.distance_transform_edt(final_mask)
    coords = peak_local_max(distance, footprint=np.ones((15, 15)), labels=final_mask)

    markers = np.zeros_like(distance, dtype=int)
    markers[tuple(coords.T)] = np.arange(len(coords)) + 1

    segmented = watershed(-distance, markers, mask=final_mask)
    labeled_filtered = label(segmented)


    ### Final analysis ###

    props_final = regionprops_table(labeled_filtered, properties=["label", "area"])
    df_final = pd.DataFrame(props_final)
    df_final["area_mm2"] = df_final["area"] / PP_SQMM

    final_count = len(df_final)
    mean_area = df_final["area_mm2"].mean()                                       # Finding average size of connected components


    ### Output for User ##
    
    print(f"Filepath: {path}")
    print(f"Total number of unfltred seeds: {len(df)}")                             # Unfiltered = all connected components (Seeds) of area > .1 mm^2
    print(f"Total number of filtered seeds: {final_count}")                    # Filtered = all connected components (seeds) of area > .1 mm^2 & is within 80% of data that is closest to mean
    print(f"Average seed size is: {mean_area}")
    print()


    ### Histogram ###
    
    sns.histplot(df_final["area_mm2"], bins=50)                                    # Histogram for filtered seeds (basically 1.25 mm^2 - 2.5 mm^2)
    plt.title("Seed Area Distribution")
    plt.xlabel("Area (mm^2)")
    plt.show()


    return final_count, mean_area


def Cycle(folder = "Data"):
    folder_path = Path(folder)
    tif_files = [file for file in folder_path.glob("*.tif")]

    for tif_file in tif_files:
        Run(tif_file.name)

Cycle("Data")
