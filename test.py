import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import transform

arr = np.asarray([0,0,0,255,255,128,255,255,255])
arr = np.transpose(transform.resize(arr, (arr.shape[0] + 5, 1), preserve_range=True, anti_aliasing=False))

plt.imshow(arr)
plt.show()