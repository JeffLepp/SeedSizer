
This program was designed to analyze .tif flatbed scans of caminilla seeds behind a black surface, 
if you're images differ from this there may be unexpected results, but should work if:

    1. You have brighter objects behind a dark surface
    2. Touching objects are limited as much as possible before scanning
    3. Using a high resolution image such as .TIF, .BMP


## How to run

1: Install dependencies with command in console:

    'pip install -r requirements.txt'

    NOTE: if this fails open requirements.txt and install each manually with:

        'pip install <DEPENDENCY>', 
        do not include the version number with this: numpy>=1.24 would be 'pip install numpy'

2: Navigate to SEEDSIZE directory in console and run 'python SeedSizer.py'

3: Select the folder containing batch of .tiff file images

Expected Result: Updated output.csv which shows all .tiff files results sequentially. 
Note: Results may take some time to initially print


## This Python program analyzes high-resolution seed scan images (TIFF format) to calculate:

- Total number of seeds
- Size of each seed in scan
- Total size of seed
- General size analytics

## Features

- Reads large `.tif` seed images
- Applies Otsu's threshold to distinguish seeds from the background
- Filters out small objects (noise)
- Outputs seed areas in a pandas DataFrame
- calculates seed count, average size, total size, and other analytics

## Output 

File
 Name of the analyzed TIFF image. Each file corresponds to one scanned batch of seeds.

Seed_Count
 Total number of detected seeds (objects) after removing small particles and noise. Represents the estimated seed count in the image.

Mean_Area_mm2
 Average projected area of individual seeds, expressed in square millimeters (mm²). Reflects the mean seed size within the batch.

Median_Area_mm2
 Median (middle) seed area, providing a robust estimate of central tendency that is less influenced by extreme sizes.

StdDev_Area_mm2
 Standard deviation of seed area measurements, describing how much seed size varies within the image.

Var_Area_mm2
 Variance of seed area (the square of the standard deviation). Quantifies total variability in seed size.

Mean_Eccentricity
 Average elongation of seeds. A value of 0 indicates a perfect circle; values approaching 1 represent more oval or elongated shapes.

StdDev_Eccentricity
 Standard deviation of seed eccentricity, showing how consistent the seed shapes are. Low values mean uniform shape; high values indicate irregularity.

Mean_Solidity
 Average compactness of seeds, defined as the ratio of seed area to its convex hull area. Values near 1 suggest smooth, convex seeds; lower values indicate rough or irregular outlines.

Mean_Aspect_Ratio
 Average ratio of each seed’s major axis to minor axis length. Values greater than 1 indicate elongated seeds.

Total_Area_mm2
 Combined area of all detected seeds (sum of individual seed areas), used as an approximate indicator of total yield or total seed mass in the image.