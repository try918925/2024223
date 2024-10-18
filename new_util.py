import time
import cv2
import numpy as np
from scipy.stats import norm

# trapezoid_r = np.array([[450, 637], [2120, 684], [456, 880], [2118, 811]], dtype=np.float32)
# rectangle_r = np.array([[200, 0], [1040, 0], [200, 300], [1040, 300]], dtype=np.float32)
# self.M_right = cv2.getPerspectiveTransform(trapezoid_r, rectangle_r)
#
# trapezoid_l = np.array([[392, 635], [2130, 560], [393, 795], [2126, 872]], dtype=np.float32)
# rectangle_l = np.array([[200, 0], [1040, 0], [200, 300], [1040, 300]], dtype=np.float32)
# self.M_left = cv2.getPerspectiveTransform(trapezoid_l, rectangle_l)
#
# self.distortion_coeffs = np.array([-7.7041889e+00, 1.909734151e+00,  -2.68401705e-03,  \
#                             8.63032028e-02,-1.07008640e+02])
# self.camera_matrix = np.array([[9.73243587e+03, 0.00000000e+00, 1.22599216e+03],
#                         [0.00000000e+00, 9.58544217e+03, 7.26588013e+02],
#                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


trapezoid_r = np.array([[450, 637], [2120, 684], [456, 880], [2118, 811]], dtype=np.float32)
rectangle_r = np.array([[200, 0], [1040, 0], [200, 300], [1040, 300]], dtype=np.float32)
M_right = cv2.getPerspectiveTransform(trapezoid_r, rectangle_r)

trapezoid_l = np.array([[392, 635], [2130, 560], [393, 795], [2126, 872]], dtype=np.float32)
rectangle_l = np.array([[200, 0], [1040, 0], [200, 300], [1040, 300]], dtype=np.float32)
M_left = cv2.getPerspectiveTransform(trapezoid_l, rectangle_l)

distortion_coeffs = np.array([-7.7041889e+00, 1.909734151e+00, -2.68401705e-03, \
                              8.63032028e-02, -1.07008640e+02])
camera_matrix = np.array([[9.73243587e+03, 0.00000000e+00, 1.22599216e+03],
                          [0.00000000e+00, 9.58544217e+03, 7.26588013e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


def crop_images(img, v):
    start_time = time.time()
    if v == 'right':
        result = cv2.warpPerspective(img, M_right, (2560, 1440))
        result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
        return result[130:1220, 940:1540]
    elif v == 'left':
        result = cv2.warpPerspective(img, M_left, (2560, 1440))
        result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return result[1450:2540, 0:300]
    elif v == 'top':
        undistorted_img = cv2.undistort(img, camera_matrix, distortion_coeffs)
        return cv2.rotate(undistorted_img, cv2.ROTATE_90_CLOCKWISE)

    print("crop_images(self, img, v):", time.time() - start_time)


class crop_img(object):
    # def __init__(self) -> None:
    #     pass
    # 直接存全尺寸的frame进lst就行
    def crop_images(self, img, v):
        start_time = time.time()
        if v == 'right':
            result = cv2.warpPerspective(img, M_right, (2560, 1440))
            result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
            return result[130:1220, 940:1540]
        elif v == 'left':
            result = cv2.warpPerspective(img, M_left, (2560, 1440))
            result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return result[1450:2540, 0:300]
        elif v == 'top':
            undistorted_img = cv2.undistort(img, camera_matrix, distortion_coeffs)
            return cv2.rotate(undistorted_img, cv2.ROTATE_90_CLOCKWISE)

        print("crop_images(self, img, v):", time.time() - start_time)


# def crop_images(img, v):
#     if v == 'right':
#         trapezoid = np.array([[450, 637], [2120, 684], [456, 880], [2118, 811]], dtype=np.float32)
#         rectangle = np.array([[200, 0], [1040, 0], [200, 300], [1040, 300]], dtype=np.float32)
#         # 计算透视变换矩阵
#         M = cv2.getPerspectiveTransform(trapezoid, rectangle)
#         # 进行透视变换
#         result = cv2.warpPerspective(img, M, (2560, 1440))
#         result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
#
#         result = result[130:1220, 940:1540]
#
#     elif v == 'left':
#         trapezoid = np.array([[392, 635], [2130, 560], [393, 795], [2126, 872]], dtype=np.float32)
#         rectangle = np.array([[200, 0], [1040, 0], [200, 300], [1040, 300]], dtype=np.float32)
#         # 计算透视变换矩阵
#         M = cv2.getPerspectiveTransform(trapezoid, rectangle)
#         # 进行透视变换
#         result = cv2.warpPerspective(img, M, (2560, 1440))
#         result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
#
#         result = result[1450:2540, 0:300]
#
#     elif v == 'top':
#         distortion_coeffs = np.array([-7.7041889e+00, 1.909734151e+00,  -2.68401705e-03,  \
#                                     8.63032028e-02,-1.07008640e+02])
#         camera_matrix = np.array([[9.73243587e+03, 0.00000000e+00, 1.22599216e+03],
#                                 [0.00000000e+00, 9.58544217e+03, 7.26588013e+02],
#                                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
#         undistorted_img = cv2.undistort(img, camera_matrix, distortion_coeffs)
#
#         result = cv2.rotate(undistorted_img, cv2.ROTATE_90_CLOCKWISE)


def concateImg(img1, img2, off, v):
    if v == 'right':
        height, width = img1.shape[:2]
        blank_image = np.zeros((height, width + off, 3), dtype=np.uint8)
        if off > img2.shape[1]:
            stitchImg_part = img2
            blank_image[:, blank_image.shape[1] - img1.shape[1]:blank_image.shape[1]] = img1
            blank_image[:, 0:blank_image.shape[1] - img2.shape[1]] = stitchImg_part

        else:
            stitchImg_part = img2[:, 0:off]
            blank_image[:, blank_image.shape[1] - img1.shape[1]:blank_image.shape[1]] = img1
            blank_image[:, 0:blank_image.shape[1] - img1.shape[1]] = stitchImg_part

    elif v == 'left':
        height, width = img1.shape[:2]
        blank_image = np.zeros((height, width + off, 3), dtype=np.uint8)
        if off > img2.shape[1]:
            stitchImg_part = img2
            blank_image[:, 0:img1.shape[1]] = img1
            blank_image[:, blank_image.shape[1] - img2.shape[1]:blank_image.shape[1]] = stitchImg_part
        else:
            stitchImg_part = img2[:, img2.shape[1] - off:img2.shape[1]]
            blank_image[:, 0:img1.shape[1]] = img1
            blank_image[:, blank_image.shape[1] - off:blank_image.shape[1]] = stitchImg_part
    elif v == 'top':
        height, width = img1.shape[:2]
        blank_image = np.zeros((height, width + off, 3), dtype=np.uint8)
        if off > img2.shape[1]:
            stitchImg_part = img2
            blank_image[:, 0:img1.shape[1]] = img1
            blank_image[:, blank_image.shape[1] - img2.shape[1]:blank_image.shape[1]] = stitchImg_part
        else:
            stitchImg_part = img2[:, img2.shape[1] - off:img2.shape[1]]
            blank_image[:, 0:img1.shape[1]] = img1
            blank_image[:, blank_image.shape[1] - off:blank_image.shape[1]] = stitchImg_part

    return blank_image


def rectifyOff(offLst):
    # print(offLst)
    non_zero_values = [val for val in offLst if val != 0]

    if non_zero_values:
        mean_value = sum(non_zero_values) / len(non_zero_values)
        return int(mean_value)
    else:
        return 0


def stichAll(imgLst, offLst, v):
    temp = None
    # lastNonZero = getLastNonZero(offLst)
    rectify_off = rectifyOff(offLst)

    for i in range(len(imgLst)):
        if i > 1:
            img2 = imgLst[i]
            off = offLst[i - 1]
            if off > 1:
                temp = concateImg(temp, img2, off, v)

            # elif off == 0 and i <= lastNonZero:
            elif off <= 1:
                temp = concateImg(temp, img2, rectify_off, v)
        else:
            temp = imgLst[i]

    return temp


def estimate_translation_orb(img1, img2, v):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if v == 'right':
        img1 = img1[100:800]
        img2 = img2[100:800]
        # nfeatures: 要保留的特征点数目。默认为 500。
        # scaleFactor: 图像金字塔中每一层的缩放系数。默认为 1.2。
        # nlevels: 金字塔的层数。默认为 8。
        # edgeThreshold: 检测特征点时的边界阈值，较大的值会避免特征点接近边界处。默认为 31。
        # firstLevel: 第一层金字塔的索引。默认为 0。
        # WTA_K: 用于描述符的比较位数。可以是 2 或 4，影响描述符的长度。默认为 2。
        # scoreType: 特征点的评分算法。可以是 cv2.ORB_HARRIS_SCORE 或 cv2.ORB_FAST_SCORE，默认为 HARRIS_SCORE。
        # patchSize: 用于计算描述符的特征点邻域大小。默认为 31。
        # fastThreshold: FAST 算法的阈值。这个参数会影响初始的特征点检测速度和准确度。默认为 20。
        orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.1301, nlevels=7, edgeThreshold=9,
                             firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

        if descriptors1 is not None and descriptors2 is not None:
            matches = bf.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)

            points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
            points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

            valid_points1 = []
            valid_points2 = []

            diff_p = np.array(points2) - np.array(points1)

            sort_diff_p = sorted(diff_p[:, :1])
            mu_x_p, _ = norm.fit(sort_diff_p[int(0.2 * len(sort_diff_p)):int(0.8 * (len(sort_diff_p)))])
            for i in range(len(points1)):
                if points2[i][0] - points1[i][0] > mu_x_p and (points2[i][1] - points1[i][1]) < 10:
                    valid_points1.append(points1[i])
                    valid_points2.append(points2[i])

            if len(valid_points1) == 0:
                return 0

            diff = np.array(valid_points2) - np.array(valid_points1)

            sort_diff = sorted(diff[:, :1])
            if len(sort_diff) > 3:
                mu_x, std_x = norm.fit(sort_diff[int(0.25 * len(sort_diff)):int(0.75 * (len(sort_diff)))])
            else:
                mu_x, std_x = norm.fit(sort_diff)
            # translation = np.mean(np.array(valid_points2) - np.array(valid_points1), axis=0)

            return round(abs(mu_x))
        else:
            return 0
    elif v == 'left':
        # print('===========')
        t1 = time.time()
        img1 = img1[100:900]
        img2 = img2[100:900]
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

        if descriptors1 is not None and descriptors2 is not None:
            matches = bf.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)

            points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
            points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

            valid_points1 = []
            valid_points2 = []

            for i in range(len(points1)):
                if points2[i][0] - points1[i][0] < -1:
                    valid_points1.append(points1[i])
                    valid_points2.append(points2[i])

            if len(valid_points1) == 0:
                return 0

            diff = np.array(valid_points2) - np.array(valid_points1)
            sort_diff = sorted(diff[:, :1])
            if len(sort_diff) > 3:
                mu_x, std_x = norm.fit(sort_diff[int(0.25 * len(sort_diff)):int(0.75 * (len(sort_diff)))])
            else:
                mu_x, std_x = norm.fit(sort_diff)
            # translation = np.mean(np.array(valid_points2) - np.array(valid_points1), axis=0)
            # print(time.time() - t1)
            # print('======================')
            return round(abs(mu_x))
        else:
            return 0

    elif v == 'top':
        orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.1001, nlevels=7, edgeThreshold=21,
                             firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

        if descriptors1 is not None and descriptors2 is not None:
            matches = bf.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)

            points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
            points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

            valid_points1 = []
            valid_points2 = []

            for i in range(len(points1)):
                if points2[i][0] - points1[i][0] < -10 and (points2[i][1] - points1[i][1]) < 10:
                    valid_points1.append(points1[i])
                    valid_points2.append(points2[i])

            if len(valid_points1) == 0:
                return 0

            diff = np.array(valid_points2) - np.array(valid_points1)

            sort_diff = sorted(diff[:, :1])
            if len(sort_diff) > 3:
                mu_x, std_x = norm.fit(sort_diff[int(0.25 * len(sort_diff)):int(0.75 * (len(sort_diff)))])
            else:
                mu_x, std_x = norm.fit(sort_diff)
            # translation = np.mean(np.array(valid_points2) - np.array(valid_points1), axis=0)

            return round(abs(mu_x))

        else:

            return 0
