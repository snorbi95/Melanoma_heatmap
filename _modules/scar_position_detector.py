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

def scar_midpoint_detection(original_image_path):
    original_image = io.imread(original_image_path)
    gray_image = color.rgb2gray(original_image)
    # gray_image = filters.gaussian(gray_image, sigma = 2)

    # save hsv
    hsv_image = color.rgb2hsv(original_image)
    # fig, ax = plt.subplots(1,3)
    # for i in range(3):
    #     ax[i].imshow(hsv_image[:,:,i])
    # plt.show()
    # plt.savefig(f'hsv_figures/{image_name}', dpi = 300)
    gray_image = exposure.equalize_hist(hsv_image[:, :, 1])
    mean_of_hsv = np.mean(hsv_image[:, :, 2])
    diff_image = np.abs(gray_image - mean_of_hsv) ** 3
    diff_image = (hsv_image[:, :, 1] * 2) - diff_image
    #diff_image = transform.rescale(diff_image, 0.5)
    mid_point = detect_scars(diff_image)
    return mid_point

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

def detect_scars(in_image):
    mask_size = 25
    filter_threshold = 0.7
    res_image = np.zeros_like(in_image)

    grid = np.indices((res_image.shape[0], res_image.shape[1]))
    grid[0] = (res_image.shape[0] // 2) - np.abs(grid[0] - (res_image.shape[0] // 2))
    grid[1] = (res_image.shape[1] // 2) - np.abs(grid[1] - (res_image.shape[1] // 2))

    grid_mtx = ((grid[0] + grid[1]) // 2) ** 2
    grid_mtx = grid_mtx.astype(np.float64)
    grid_mtx *= 1 / grid_mtx.max()

    in_image = grid_mtx - in_image

    # plt.imshow(in_image)
    # plt.show()

    while True:
    #res_image[in_image > 0.35] = 1
        for i in range(mask_size // 2, in_image.shape[0] - mask_size // 2, mask_size // 2):
            for j in range(mask_size // 2, in_image.shape[1] - mask_size // 2, mask_size // 2):
                act_mtx = in_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                              j - (mask_size // 2):j + 1 + (mask_size // 2)]
                # act_vec = np.reshape(act_mtx, (mask_size * mask_size, 1))
                # number_of_lows = act_vec[act_vec > 0.75].size
                # if number_of_lows > len(act_vec) * 0.5:
                if np.mean(act_mtx) > filter_threshold:
                    res_image[i - (mask_size // 2):i + 1 + (mask_size // 2),
                              j - (mask_size // 2):j + 1 + (mask_size // 2)] = 1

        # plt.imshow(res_image)
        # plt.show()

        seeds = (np.where(res_image == 1)[0], np.where(res_image == 1)[1])
        # print(seeds[0].shape)

        if len(seeds[0]) != 0:
            start_x = np.min(seeds[0])
            start_y = np.min(seeds[1])
            end_x = np.max(seeds[0])
            end_y = np.max(seeds[1])
            mid_point = ((start_y + end_y) // 2,(start_x + end_x) // 2)
            break
        else:
            filter_threshold = filter_threshold - 0.25
            # mid_point = (res_image.shape[1] // 2, res_image.shape[0] // 2)
            # break

    #mid_point_ratio = (mid_point[0] / res_image.shape[1], mid_point[1] / res_image.shape[0])
    # mid_point_mtx = np.zeros((4,1,2))
    # # for i in range(4):
    # #     mid_point_mtx[i] = mid_point
    # mid_point_mtx[0] = mid_point
    # mid_point_mtx[1] = (mid_point[0] + 1, mid_point[1])
    # mid_point_mtx[2] = (mid_point[0], mid_point[1] + 1)
    # mid_point_mtx[3] = (mid_point[0] + 1, mid_point[1] + 1)
    #print(res_image.shape)
    return mid_point


# body_images = [f for f in listdir('../_images/body_images_resized') if isfile(join('../_images/body_images_resized', f))]
# scar_images = [f for f in listdir('../_images/scar_images_resized') if isfile(join('../_images/scar_images_resized', f))]
#
# for image_name in scar_images:
#     #open a scar image
#     print(f'Processing {image_name}')
#     original_image = io.imread(f'../_images/scar_images_resized/{image_name}')
#
#     gray_image = color.rgb2gray(original_image)
#     #graüy_image = filters.gaussian(gray_image, sigma = 2)
#
#     #save hsv
#     hsv_image = color.rgb2hsv(original_image)
#     # fig, ax = plt.subplots(1,3)
#     # for i in range(3):
#     #     ax[i].imshow(hsv_image[:,:,i])
#     # plt.show()
#     # plt.savefig(f'hsv_figures/{image_name}', dpi = 300)
#     gray_image = exposure.equalize_hist(hsv_image[:,:,1])
#     mean_of_hsv = np.mean(hsv_image[:,:,2])
#     diff_image = np.abs(gray_image - mean_of_hsv) ** 3
#     diff_image = (hsv_image[:,:,1] * 2) - diff_image
#     plt.imsave(f'../_images/diff_images/{image_name}', diff_image, cmap = 'gray')
#     # plt.imshow(diff_image)
#     # plt.show()
#
#     mid_point = detect_scars(diff_image)
#     # print(mid_point)
#     mid_circle = Circle(mid_point, 50)
#     # plt.imshow(original_image)
#     # plt.plot(mid_point[0], mid_point[1], 'r', linewidth = 2, markersize = 12)
#     # plt.show()
#
#     fig, ax = plt.subplots(1)
#     ax.imshow(original_image)
#     ax.add_patch(mid_circle)
#     plt.savefig(f'../_images/scar_images_detected/{image_name}')
#     plt.clf()
