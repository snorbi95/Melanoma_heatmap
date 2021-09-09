from os import listdir
from os.path import isfile, join

import numpy as np
import sklearn.preprocessing
from numpy.f2py.auxfuncs import throw_error
from skimage import filters, io, color, feature, exposure, segmentation, transform, measure, morphology, registration
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn import cluster
from scipy import ndimage as ndi
import cv2


def adjust_image_to_reference(in_image, subtract_image):

    print(in_image.shape)
    print(subtract_image.shape)

    for i in range(subtract_image.shape[0]):
        subtract_vec = subtract_image[i,:]
        num_of_max = subtract_vec[subtract_vec == 127].size
        num_of_lows = subtract_vec[(subtract_vec == 193)].size
        # print(num_of_max)
        if num_of_max == 0:
            new_img_vec = transform.resize(in_image[i,:], (in_image.shape[1] - num_of_lows,), preserve_range=True,
                                           anti_aliasing=False)
            #print(new_img_vec.shape[0], in_image[i, :].shape[0])
            zeros_vec = np.zeros((in_image.shape[1] - new_img_vec.shape[0],)).astype(np.uint32)
            #print(zeros_vec)
            # new_vec = np.concatenate(zeros_vec, new_img_vec)
            # print(new_vec)
            in_image[i,:] = np.concatenate((zeros_vec, new_img_vec))
        else:
            new_img_vec = transform.resize(in_image[i,:], (in_image.shape[1] + num_of_max,), preserve_range=True, anti_aliasing=False)
            #print(new_img_vec.shape[0], in_image[i,:].shape[0])
            in_image[i, :] = new_img_vec[new_img_vec.shape[0] - in_image[i,:].shape[0]:]
    return in_image


if __name__ == "__main__":

    from os import listdir
    from os.path import isfile, join

    body_part = 'back'
    reference_image_path = f'_images/reference_body_images/{body_part}.jpg'
    contour_images = [f for f in listdir(f'_images/body_contour_images/{body_part}') if isfile(join(f'_images/body_contour_images/{body_part}', f))]
    print(len(contour_images))

    # read images
    reference_image = cv2.imread(reference_image_path, 0)
    inverted_reference_image = np.invert(reference_image)

    for i in range(len(contour_images)):
        current_image_name = contour_images[i]
        moving_image = cv2.imread(f'_images/body_contour_images/{body_part}/{contour_images[i]}',0)
        moving_image = cv2.resize(moving_image, (reference_image.shape[1], reference_image.shape[0]), interpolation=cv2.INTER_AREA)

        print(reference_image.shape)
        print(moving_image.shape)
        subtracted_image = (reference_image // 2) - (moving_image // 4)

        contour_with_point_image = cv2.imread(f'_images/body_contour_images/{contour_images[i]}',0)
        contour_with_point_image = cv2.resize(contour_with_point_image, (reference_image.shape[1], reference_image.shape[0]), interpolation=cv2.INTER_AREA)

        contour_with_point_image_copy = np.copy(contour_with_point_image)

        # fig, ax = plt.subplots(1,3)
        # ax[0].imshow(moving_image)
        # ax[1].imshow(reference_image)
        # ax[2].imshow(contour_with_point_image)
        # plt.show()

        left_side_img = adjust_image_to_reference(contour_with_point_image[:,:subtracted_image.shape[1] // 2], subtracted_image[:,:subtracted_image.shape[1] // 2])
        # plt.imshow(subtracted_image)
        # plt.show()
        rotated_subtracted_image = np.flip(subtracted_image[:,subtracted_image.shape[1] // 2:], axis = 1)
        rotated_moving_image = np.flip(contour_with_point_image[:, subtracted_image.shape[1] // 2:], axis=1)

        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(rotated_moving_image)
        # ax[1].imshow(rotated_subtracted_image)
        # plt.show()

        right_side_img = np.flip(adjust_image_to_reference(rotated_moving_image, rotated_subtracted_image), axis = 1)

        inverted_reference_image[contour_with_point_image == 76] += 20

        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(rotated_subtracted_image)
        # ax[1].imshow(rotated_moving_image)
        #
        # plt.show()
        # adjusted_contour_image = np.zeros_like(contour_with_point_image)
        # adjusted_contour_image[:,:subtracted_image.shape[1] // 2] = left_side_img
        # adjusted_contour_image[:, subtracted_image.shape[1] // 2:] = right_side_img

        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(contour_with_point_image_copy)
        # ax[1].imshow(contour_with_point_image)
        #
        # plt.show()

        # shifts, error, phasediff = registration.phase_cross_correlation(reference_image, moving_image)
        #
        # print(shifts, error, phasediff)
    plt.imsave(f'_images/reference_body_images/{body_part}_heatmap.jpg',np.invert(inverted_reference_image), cmap = 'gray')