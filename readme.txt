Wilson's Seed Counter Algorithm
Written by Jefferson Kline, for assistance email jefferson.kline@wsu.edu

Currently Being Developed as of 5/22/25

## Before running

This program was designed to analyze .tif flatbed scans of caminilla seeds behind a black surface, if you're images differ from this there may be unexpected results, but should work if:

    1. You have brighter objects behind a dark surface
    2. Touching objects are limited as much as possible before scanning
    3. Using a high resolution image such as .TIF, .BMP


## How to run

1: Install dependencies with command in console:

    'pip install -r requirements.txt'

    NOTE: if this fails open requirements.txt and install each manually with:

        'pip install <DEPENDENCY>', do not include the version number with this: numpy>=1.24 would be 'pip install numpy'


2: Update 'SEEDSIZE/Data' folder to your set of seed scanner images that are .tif

3: Navigate to SEEDSIZE directory in console and run 'python SeedSizer.py'

Expected Result: Updated output.csv which shows all .tiff files results sequentially. 
Note: Results may take some time to initially print


## This Python program analyzes high-resolution seed scan images (TIFF format) to calculate:

- Total number of seeds
- Size of each seed in scan
- Total size of seed


## Features

- Reads large `.tif` seed images
- Applies Otsu's threshold to distinguish seeds from the background
- Filters out small objects (noise)
- Outputs seed areas in a pandas DataFrame
- calculates seed count, average size, and total size
