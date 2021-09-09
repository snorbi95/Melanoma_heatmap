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

def km_clust(array, n_clusters):
    X = array.reshape((-1, 1))
    k_m = cluster.KMeans(n_clusters=n_clusters, n_init=4)
    k_m.fit(X)
    #parameters.wcss.append(k_m.inertia_)
    values = k_m.cluster_centers_.squeeze()
    labels = k_m.labels_
    # silhouette_avg = metrics.silhouette_score(X, labels)
    # sample_silhouette_values = metrics.silhouette_samples(X, labels)

    return(values, labels)

def get_cluster(k,img):
    values, labels = km_clust(img, n_clusters=k)
    res = np.choose(labels, values)
    res.shape, labels.shape = img.shape, img.shape
    #res[np.where(res != res.max())] = 0
    return res

def get_max_len_list(li):
    # max_len = 0
    # max_list = []
    # for item in li:
    #     if item.shape[0] > max_len:
    #         max_len = item.shape[0]
    #         max_list = item
    # return max_list
    len_and_list = [(item.shape[0], item) for item in li]
    len_and_list = sorted(len_and_list, key=lambda x:x[0], reverse=True)
    return len_and_list[0]


def get_body_contour(image_path, mid_point = None):
    print(f'Processing {image_path}')
    original_image = io.imread(image_path)

    gray_image = color.rgb2gray(original_image)
    hsv_image = color.rgb2hsv(original_image)
    # fig, ax = plt.subplots(1,3)
    # for i in range(3):
    #     ax[i].imshow(hsv_image[:,:,i])
    # plt.show()
    filtered_image = filters.gaussian(hsv_image[:, :, 0], sigma=2)
    threshold = filters.threshold_mean(hsv_image[:, :, 0])

    if threshold > 0.1:
        gray_image[(hsv_image[:,:,0] > 0.1) & (hsv_image[:,:,0] < 0.9)] = 0
        gray_image[gray_image != 0] = 1
        gray_image = ndi.binary_fill_holes(gray_image)
    else:
        # fig, ax = plt.subplots(1, 3)
        # for i in range(3):
        #     ax[i].imshow(hsv_image[:, :, i])
        # plt.show()
        threshold_1 = filters.threshold_mean(hsv_image[:,:,1])
        print(threshold_1)
        gray_image[hsv_image[:,:,1] < 0.3] = 0
        gray_image[gray_image != 0] = 1
        gray_image = ndi.binary_fill_holes(gray_image)

    res_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3))

    #gray_image = morphology.area_opening(gray_image)

    edge_image = feature.canny(gray_image)
    #
    # plt.imshow(edge_image)
    # plt.show()

    res_image[gray_image == 1] = [1,1,1]
    if mid_point != None:
        mid_point = (mid_point[0], mid_point[1])
        # print(mid_point)
        # print(res_image.shape)
        plt.imsave(f'_images/body_contour_images/blank_contours/{image_path.split("/")[-1]}', res_image)
        res_image = cv2.circle(res_image, mid_point, 25, (1, 0, 0), -1)
    #plt.savefig(f'body_contour_images/{image_path.split("/")[1]}')
    plt.imsave(f'_images/body_contour_images/{image_path.split("/")[-1]}', res_image)
    #plt.clf()

# file_path = f'../_images/body_images_resized'
# body_images = [f for f in listdir(file_path) if isfile(join(file_path, f))]
#
# for image_name in body_images:
#     get_body_contour(f'{file_path}/{image_name}')


