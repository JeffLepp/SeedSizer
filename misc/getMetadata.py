# Replace file on line 4 with the file you want to analyze

import tifffile as tiff
file = 'Data/Calena-18.tif'

with tiff.TiffFile(file) as tif:
    for tag in tif.pages[0].tags.values():
        print(tag.name, tag.value)

# Output for Data/Calena-18.tif:
'''
NewSubfileType 0
ImageWidth 10200
ImageLength 14040
BitsPerSample (8, 8, 8)
Compression 1
PhotometricInterpretation 2
StripOffsets (240,)
Orientation 1
SamplesPerPixel 3
RowsPerStrip 14040
StripByteCounts (429624000,)
MinSampleValue (0, 0, 0)
MaxSampleValue (255, 255, 255)
XResolution (1200, 1)
YResolution (1200, 1)
ResolutionUnit 2
'''

# Notes explaining imports used, high level:

import tifffile as tiff
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

# IMREAD:
#   imread is a function from the skimage library that reads an image from a file and returns it as a NumPy 2 x 2 array.

# SKIIMAGE:
#   skimage is a Python library for image processing, we will use it to make the image grayscale.
#   use otsu thershholding, which will find the best value to ditinguish the seeds from the background.
#   use measure.label to label connected regions by turning seeds into binary images.
#   use regionprops to compute parameters of our labeled seeds. 
#   use skimage.morphology to remove small artifacts and splitting overlapping seeds.
