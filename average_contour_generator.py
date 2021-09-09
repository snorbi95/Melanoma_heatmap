from os import listdir
from os.path import isfile, join

import numpy as np
import sklearn.preprocessing
from numpy.f2py.auxfuncs import throw_error
from skimage import filters, io, color, feature, exposure, segmentation, transform, measure, morphology
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn import cluster
from scipy import ndimage as ndi
import cv2

body_part = 'back'
file_path = f'_images/body_contour_images/{body_part}'

contour_images = [f for f in listdir(file_path) if isfile(join(file_path, f))]

N = 0
avg_width = 0
avg_height = 0

def extend_lines(in_image, extension_area = 25):
    for j in range(in_image.shape[1]):
        for i in range(in_image.shape[0] - 1, in_image.shape[0] - extension_area, -1):
            if in_image[i,j] != 0:
                in_image[i: in_image.shape[0], j] = 1

    for j in range(in_image.shape[1]):
        for i in range(0, extension_area):
            if in_image[i,j] != 0:
                in_image[:i, j] = 1

    for i in range(in_image.shape[1] - 1, in_image.shape[1] - extension_area, -1):
        for j in range(in_image.shape[0]):
            print(i,j)
            if in_image[j,i] != 0:
                in_image[j, i: in_image.shape[1]] = 1

    # plt.imshow(in_image)
    # plt.show()
    return in_image


for image_name in contour_images:
    original_image = io.imread(f'{file_path}/{image_name}')
    avg_width += original_image.shape[1]
    avg_height += original_image.shape[0]
    N += 1

avg_width = int(avg_width / N)
avg_height = int(avg_height / N)

avg_img = np.zeros((avg_height, avg_width, 3))

for image_name in contour_images:
    original_image = io.imread(f'{file_path}/{image_name}')
    dim = (avg_width, avg_height)

    original_image = cv2.resize(original_image, dim, interpolation=cv2.INTER_AREA)
    avg_img += original_image

avg_img = (avg_img / N).astype(np.uint8)
gray_avg_image = color.rgb2gray(avg_img)
#gray_avg_image[(gray_avg_image < 0.5) | (gray_avg_image > 0.9)] = 0
gray_avg_image[gray_avg_image == np.max(gray_avg_image)] = 0
gray_avg_image[gray_avg_image != 0] = 1

gray_avg_image = gray_avg_image.astype(np.uint8)
gray_avg_image = morphology.dilation(gray_avg_image, selem = morphology.disk(10))

plt.imshow(gray_avg_image)
plt.show()

skeleton = morphology.thin(gray_avg_image)

#skeleton = ndi.binary_fill_holes(skeleton)

plt.imshow(skeleton)
plt.show()

skeleton = extend_lines(skeleton, 50 )
skeleton = skeleton.astype(np.uint32)
skeleton = morphology.dilation(skeleton, selem = morphology.disk(3))

plt.imshow(skeleton)
plt.show()

print((skeleton.shape[0] // 2, skeleton.shape[1] // 2))

filled_skeleton = morphology.flood_fill(skeleton, seed_point=(skeleton.shape[0] // 2, skeleton.shape[1] // 2), new_value=255)
filled_skeleton = morphology.closing(filled_skeleton, selem = morphology.disk(5))
plt.imshow(filled_skeleton)
plt.show()


plt.imsave(f'_images/reference_body_images/{body_part}.jpg',filled_skeleton, cmap = 'gray')