#    Wilson's seed algorithm, 5/5/25

#    To run for the first time:
#       1. From terminal enter "pip install scikit-image"
#       2. Navigate to SeedSize directory in console and run "python getImage.py"

#    X & Y Resoltuion is 1200 pixels per inch
#       ie. square inch is 1200x1200 pixels.


from skimage.io         import imread
from skimage.color      import rgb2gray
from skimage.filters    import threshold_otsu
from skimage.measure    import label, regionprops

# TODO: Replace file with analysis of every file in Data.
image = imread('Data/Calena-18.tif')
grayscale_image = rgb2gray(image)                   # Convert to grayscale
threshold_value = threshold_otsu(grayscale_image)   # Applies Otsu's thresholding, just finds threshold between seeds and background.
binary_image = grayscale_image > threshold_value    # This labels each pixel in the background as either 1 (True) or 0 (False), where 1 is the seed and 0 is the background.
labeled_image = label(binary_image)
 
'''
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.imshow(grayscale_image, cmap='gray')
plt.title("Grayscale")

plt.subplot(1, 2, 2)
plt.imshow(binary_image, cmap='gray')
plt.title("Binary")
plt.show()
'''