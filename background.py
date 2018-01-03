#http://nbviewer.jupyter.org/github/flothesof/posts/blob/master/20160901_RemovingBackgroundFromImage.ipynb
from skimage import filters
from skimage import io as skio
import matplotlib.pyplot as plt

FOLDER_TRAINING_SET='/Users/dboudeau/depot/leaf/'
url = FOLDER_TRAINING_SET+'FARGESIA/resized/'+'100.jpg'
img = skio.imread(url)
print("shape of image: {}".format(img.shape))
print("dtype of image: {}".format(img.dtype))
#plt.imshow(img)

sobel = filters.sobel(img)

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.dpi'] = 200

plt.imshow(sobel)