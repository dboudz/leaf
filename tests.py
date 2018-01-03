import numpy as np
from skimage import data
coins = data.coins()
histo = np.histogram(coins, bins=np.arange(0, 256))