from os import listdir
from os.path import isfile, join

import numpy as np
import sklearn.preprocessing
from numpy.f2py.auxfuncs import throw_error
from skimage import filters, io, color, feature, exposure, segmentation, transform
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn import cluster
from scipy import ndimage as ndi
import cv2

body_images = [f for f in listdir('../_images/body_images') if isfile(join('../_images/body_images', f))]
scar_images = [f for f in listdir('../_images/scar_images') if isfile(join('../_images/scar_images', f))]

for image_name in scar_images:
    original_image = io.imread(f'../_images/scar_images/{image_name}')
    r = int(original_image.shape[1] * 0.3) / original_image.shape[0]
    dim = (int(original_image.shape[1] * r), int(original_image.shape[1] * 0.3))

    original_image = cv2.resize(original_image, dim, interpolation=cv2.INTER_AREA)
    plt.imsave(f'../_images/scar_images_resized/{image_name}', original_image)

for image_name in body_images:
    original_image = io.imread(f'../_images/body_images/{image_name}')
    r = int(original_image.shape[1] * 0.3) / original_image.shape[0]
    dim = (int(original_image.shape[1] * r), int(original_image.shape[1] * 0.3))

    original_image = cv2.resize(original_image, dim, interpolation=cv2.INTER_AREA)

    plt.imsave(f'../_images/body_images_resized/{image_name}', original_image)