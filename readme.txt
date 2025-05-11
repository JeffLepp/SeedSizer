Wilson's Seed Counter Algorithm
Currently Being Developed as of 5/5/25

## How to run

1: Install dependencies with command in console:

    'pip install -r requirements.txt'

2: Update SEEDSIZE/Data folder to your set of seed scanner images that are .tif

3: Navigate to SEEDSIZE directory in console and run 'python SeedSizer.py'

Expected Result: Filepath, total unfiltered seeds counted, total seeds once filtered, average seed size for batch for each .tif iamge in SEEDSIZE/Data folder.
Note: Results may take some time to initially print


## This Python program analyzes high-resolution seed scan images (TIFF format) to calculate:

- Total number of seeds
- Size of each seed in square 
- Histograms of filtered and unfiltered data (needs to be uncommented)


## Features

- Reads large `.tif` seed images
- Applies Otsu's threshold to distinguish seeds from the background
- Filters out small objects (noise)
- Outputs seed areas in a pandas DataFrame
