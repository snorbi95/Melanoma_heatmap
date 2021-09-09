import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt
import os
import scar_position_detector as spd
import body_contour as bc


MIN_MATCH_COUNT = 35

# parser = argparse.ArgumentParser(description='Template matcher')
# parser.add_argument('--template', type=str, action='store',
#                     help='The image to be used as template')
# parser.add_argument('--map', type=str, action='store',
#                     help='The image to be searched in')
# parser.add_argument('--show', action='store_true',
#                     help='Shows result image')
# parser.add_argument('--save-dir', type=str, default='./',
#                     help='Directory in which you desire to save the result image')
#
# args = parser.parse_args()

current_image_name = ''
mid_point_ratio = 0


def get_matched_coordinates(temp_img, map_img):

    # initiate SIFT detector
    while True:
        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(temp_img, None)
        kp2, des2 = sift.detectAndCompute(map_img, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # find matches by knn which calculates point distance in 128 dim
        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        print(len(good))
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32(
                [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # find homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = temp_img.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                              [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)  # matched coordinates

            # x_coords = [vertex[0,0] for vertex in dst]
            # y_coords = [vertex[0,1] for vertex in dst]
            #
            # mid_point_ratio_percent_x = (mid_point_ratio[0] / 0.5)
            # mid_point_ratio_percent_y = (mid_point_ratio[1] / 0.5)
            #
            # x = (sum(x_coords) / len(dst))
            # y = (sum(y_coords) / len(dst))
            #print(mid_point_ratio)
            mid_points =  np.float32([[mid_point_ratio[0], mid_point_ratio[1]], [mid_point_ratio[0], mid_point_ratio[1]], [mid_point_ratio[0], mid_point_ratio[1]],
                              [mid_point_ratio[0], mid_point_ratio[1]]]).reshape(-1, 1, 2)
            mid_point = cv2.perspectiveTransform(mid_points, M)

            # mid_point = (((dst[0,0,0] + dst[2,0,0]) * mid_point_ratio[1]).astype(np.uint32),
            #              ((dst[0,0,1] + dst[2,0,1]) * mid_point_ratio[0]).astype(np.uint32))

            #mid_point = (mid_point[0].astype(np.uint32), mid_point[1].astype(np.uint32))

            #print(mid_point)

            map_img = cv2.polylines(
                map_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            mid = [mid_point[0,0,0].astype(np.uint32), mid_point[0,0,1].astype(np.uint32)]
            map_img = cv2.circle(map_img, mid, 15, (0,0,0), -1)
            print(map_img.shape)
            plt.imsave(f'../_images/result_images/{current_image_name}_placed.jpg',map_img, cmap='gray')
            break
        else:
            print("Not enough matches are found - %d/%d" %
                  (len(good), MIN_MATCH_COUNT))
            # matchesMask = None
            temp_img = temp_img[25:temp_img.shape[0] - 25, 50: temp_img.shape[1] - 25]
            map_img = map_img[25: map_img.shape[0] - 25, 25: map_img.shape[1] - 25]
            temp_img = cv2.equalizeHist(temp_img)
            map_img = cv2.equalizeHist(map_img)

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    # draw template and map image, matches, and keypoints
    img3 = cv2.drawMatches(temp_img, kp1, map_img, kp2,
                           good, None, **draw_params)

    # if --show argument used, then show result image
    # plt.imshow(img3, 'gray'), plt.show()

    # result image path
    cv2.imwrite(f'../_images/result_images/{current_image_name}_result.png', img3)
    return dst, mid


if __name__ == "__main__":

    from os import listdir
    from os.path import isfile, join

    body_images = [f for f in listdir('../_images/body_images_resized') if isfile(join('../_images/body_images_resized', f))]
    scar_images = [f for f in listdir('../_images/scar_images_resized') if isfile(join('../_images/scar_images_resized', f))]

    # read images

    for i in range(7):
        current_image_name = scar_images[i]
        temp_img_gray = cv2.imread(f'../_images/scar_images_resized/{scar_images[i]}', 0)
        map_img_gray = cv2.imread(f'../_images/body_images_resized/{body_images[i]}', 0)

        mid_point_ratio = spd.scar_midpoint_detection(f'../_images/scar_images_resized/{scar_images[i]}')

        # temp_img_gray_resized = cv2.resize(temp_img_gray, (int(temp_img_gray.shape[0] * 0.5), int(temp_img_gray.shape[1] * 0.5)))
        # map_img_gray_resized = cv2.resize(map_img_gray, (int(map_img_gray.shape[0] * 0.5), int(map_img_gray.shape[1] * 0.5)))

        temp_img_gray_resized = temp_img_gray
        map_img_gray_resized = map_img_gray

        # equalize histograms
        temp_img_eq = cv2.equalizeHist(temp_img_gray_resized)
        #temp_img_eq = temp_img_gray_resized
        map_img_eq = cv2.equalizeHist(map_img_gray_resized)
        #map_img_eq = map_img_gray_resized

        # calculate matched coordinates
        coords, mid_point = get_matched_coordinates(temp_img_eq, map_img_eq)
        # x_coords = [vertex[0,0].astype(np.uint32) for vertex in coords]
        # y_coords = [vertex[0,1].astype(np.uint32) for vertex in coords]
        # cropped_img = map_img_eq[min(y_coords): max(y_coords), min(x_coords): max(x_coords)]
        # plt.imshow(cropped_img)
        # plt.show()


        print(mid_point)


        bc.get_body_contour(f'../_images/body_images_resized/{body_images[i]}', mid_point)