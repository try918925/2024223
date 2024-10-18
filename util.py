import numpy as np
import cv2


# def cv_show(image):
#     cv2.namedWindow('1', cv2.WINDOW_NORMAL)  # 设置窗口属性为可调整大小
#     cv2.imshow('1', image)
#     cv2.resizeWindow('1', 300, 840)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def crop_images(img, v):
    if v == 'right':
        trapezoid = np.array([[450, 637], [2120, 684], [456, 880], [2118, 811]], dtype=np.float32)
        rectangle = np.array([[200, 0], [1040, 0], [200, 300], [1040, 300]], dtype=np.float32)
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(trapezoid, rectangle)
        # 进行透视变换
        result = cv2.warpPerspective(img, M, (2560, 1440))
        result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)

        result = result[130:1220, 1140:1440]
        

    elif v == 'left':
        trapezoid = np.array([[392, 635], [2130, 560], [393, 795], [2126, 872]], dtype=np.float32)
        rectangle = np.array([[200, 0], [1040, 0], [200, 300], [1040, 300]], dtype=np.float32)
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(trapezoid, rectangle)
        # 进行透视变换
        result = cv2.warpPerspective(img, M, (2560, 1440))
        result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
        result = result[1450:2540, 0:300]

    elif v == 'top':
        trapezoid = np.array([[224, 625], [2377, 604], [224, 890], [2377, 883]], dtype=np.float32)  # 左上、右上、左下、右下
        rectangle = np.array([[0, 0], [840, 0], [0, 200], [840, 200]], dtype=np.float32)
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(trapezoid, rectangle)
        # 进行透视变换
        result = cv2.warpPerspective(img, M, (840, 200))
        result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)

    # debug
    # cv2.imwrite(f"undistorted_{v}.jpg", undistorted_img)
    # cv2.imwrite(f"Perspective_{v}.jpg", result)

    return result


def getOffset(img1, img2, v):
    if v == 'right':
        stitchImg_part = img1[0:img1.shape[0], 0:150]

        result = cv2.matchTemplate(img2, stitchImg_part, cv2.TM_CCORR_NORMED)
        _, _, _, top_left = cv2.minMaxLoc(result)
    elif v == 'left':
        stitchImg_part = img1[0:img1.shape[0], img1.shape[1] - 150:img1.shape[1]]

        result = cv2.matchTemplate(img2, stitchImg_part, cv2.TM_CCORR_NORMED)
        _, _, _, top_left = cv2.minMaxLoc(result)
    elif v == 'top':
        stitchImg_part = img1[0:img1.shape[0], 0:100]

        result = cv2.matchTemplate(img2, stitchImg_part, cv2.TM_CCORR_NORMED)
        _, _, _, top_left = cv2.minMaxLoc(result)
    return top_left[0]


def getLastNonZero(offLst):
    if len(offLst) > 1:
        index = len(offLst) - 1
        while index >= 0:
            if offLst[index] != 0:
                break
            index -= 1
        return index
    else:
        return 0


def rectifyOff(offLst):
    non_zero_values = [val for val in offLst if val != 0]

    if non_zero_values:
        mean_value = sum(non_zero_values) / len(non_zero_values)
        return int(mean_value)
    else:
        return 0


def concateImg(img1, img2, off, v):
    if v == 'right':
        stitchImg_part = img1[0:img1.shape[0], 0:150]

        stitchPart = img2[0:stitchImg_part.shape[0], 0:off + stitchImg_part.shape[1]]

        matchResult = img2[0:stitchImg_part.shape[0], off:stitchImg_part.shape[1] + off]

        addImg = cv2.addWeighted(stitchImg_part, 0.55, matchResult, 0.45, 0)
        stitchPart[0:stitchImg_part.shape[0], off:stitchImg_part.shape[1] + off] = addImg

        img_h, img_w = img1.shape[:2]
        stitchImg_h, stitchImg_w = stitchPart.shape[:2]
        height = img_h
        width = img_w + stitchImg_w - stitchImg_part.shape[1]
        blank_image = np.zeros((height, width, 3), dtype=np.uint8)

        blank_image[0:img_h, stitchImg_w - stitchImg_part.shape[1]:blank_image.shape[1]] = img1
        blank_image[0:img_h, 0:stitchImg_w] = stitchPart

    elif v == 'left':
        stitchImg_part = img1[0:img1.shape[0], img1.shape[1] - 90:img1.shape[1]]

        stitchPart = img2[0:stitchImg_part.shape[0], off:img2.shape[1]]

        matchResult = img2[0:stitchImg_part.shape[0], off:stitchImg_part.shape[1] + off]

        addImg = cv2.addWeighted(stitchImg_part, 0.55, matchResult, 0.45, 0)
        stitchPart[0:stitchPart.shape[0], 0:stitchImg_part.shape[1]] = addImg

        img_h, img_w = img1.shape[:2]
        stitchImg_h, stitchImg_w = stitchPart.shape[:2]
        height = img_h
        width = img_w + stitchImg_w - stitchImg_part.shape[1]
        blank_image = np.zeros((height, width, 3), dtype=np.uint8)

        blank_image[0:img_h, 0:img_w] = img1
        blank_image[0:img_h, img_w - stitchImg_part.shape[1]:blank_image.shape[1]] = stitchPart

    elif v == 'top':
        stitchImg_part = img1[0:img1.shape[0], 0:60]

        stitchPart = img2[0:stitchImg_part.shape[0], 0:off + stitchImg_part.shape[1]]

        matchResult = img2[0:stitchImg_part.shape[0], off:stitchImg_part.shape[1] + off]

        addImg = cv2.addWeighted(stitchImg_part, 0.55, matchResult, 0.45, 0)
        stitchPart[0:stitchImg_part.shape[0], off:stitchImg_part.shape[1] + off] = addImg

        img_h, img_w = img1.shape[:2]
        stitchImg_h, stitchImg_w = stitchPart.shape[:2]
        height = img_h
        width = img_w + stitchImg_w - stitchImg_part.shape[1]
        blank_image = np.zeros((height, width, 3), dtype=np.uint8)

        blank_image[0:img_h, stitchImg_w - stitchImg_part.shape[1]:blank_image.shape[1]] = img1
        blank_image[0:img_h, 0:stitchImg_w] = stitchPart

    return blank_image


def stichAll(imgLst, offLst, v):
    print(offLst)
    temp = None
    lastNonZero = getLastNonZero(offLst)
    rectify_off = rectifyOff(offLst)
    
    for i in range(len(imgLst)):
        if i > 1:
            img2 = imgLst[i]
            off = offLst[i - 1]
            if off != 0:
                temp = concateImg(temp, img2, off, v)

            # elif off == 0 and i <= lastNonZero:
            elif off == 0:
                temp = concateImg(temp, img2, rectify_off, v)
        else:
            temp = imgLst[i]

    return temp


# def getOffset_old(img1, img2, v):
#     if v == 'right':
#         stitchImg_part = img1[0:img1.shape[0], 0:300]
#
#         templ = img1[0:img1.shape[0] - 400, 0:300]
#
#         result = cv2.matchTemplate(img2, templ, cv2.TM_CCORR_NORMED)
#         _, _, _, matchLoc = cv2.minMaxLoc(result)
#
#         top_left = matchLoc
#         print(top_left)
#         show_a = cv2.rectangle(img2.copy(), top_left, (top_left[0] + templ.shape[1], top_left[1] + templ.shape[0]),
#                                (0, 255, 0), 2)
#
#         height111 = max(templ.shape[0], show_a.shape[0])
#         width111 = templ.shape[1] + show_a.shape[1] + 200
#         blank_image111 = np.zeros((height111, width111, 3), dtype=np.uint8)
#
#         # 将img1和img2拼接在一起
#         blank_image111[0:templ.shape[0], 0:templ.shape[1]] = templ
#         blank_image111[0:show_a.shape[0], 200 + templ.shape[1]:] = show_a
#
#         cv_show(img1)
#         cv_show(blank_image111)
#
#         stitchPart = img2[0:stitchImg_part.shape[0], 0:top_left[0] + stitchImg_part.shape[1]]
#
#         matchResult = img2[0:stitchImg_part.shape[0], top_left[0]:stitchImg_part.shape[1] + top_left[0]]
#
#         addImg = cv2.addWeighted(stitchImg_part, 0.55, matchResult, 0.45, 0)
#         stitchPart[0:stitchImg_part.shape[0], top_left[0]:stitchImg_part.shape[1] + top_left[0]] = addImg
#
#         img_h, img_w = img1.shape[:2]
#         stitchImg_h, stitchImg_w = stitchPart.shape[:2]
#         height = img_h
#         width = img_w + stitchImg_w - stitchImg_part.shape[1]
#         blank_image = np.zeros((height, width, 3), dtype=np.uint8)
#
#         blank_image[0:img_h, stitchImg_w - stitchImg_part.shape[1]:blank_image.shape[1]] = img1
#         blank_image[0:img_h, 0:stitchImg_w] = stitchPart
#         result = blank_image
#
#     if v == 'left':
#         stitchImg_part = img1[0:img1.shape[0], img1.shape[1] - 150:img1.shape[1]]
#
#         result = cv2.matchTemplate(img2, stitchImg_part, cv2.TM_CCORR_NORMED)
#         _, _, _, matchLoc = cv2.minMaxLoc(result)
#
#         top_left = matchLoc
#
#         stitchPart = img2[0:stitchImg_part.shape[0], top_left[0]:img2.shape[1]]
#
#         matchResult = img2[0:stitchImg_part.shape[0], top_left[0]:stitchImg_part.shape[1] + top_left[0]]
#
#         addImg = cv2.addWeighted(stitchImg_part, 0.55, matchResult, 0.45, 0)
#         stitchPart[0:stitchPart.shape[0], 0:stitchImg_part.shape[1]] = addImg
#
#         img_h, img_w = img1.shape[:2]
#         stitchImg_h, stitchImg_w = stitchPart.shape[:2]
#         height = img_h
#         width = img_w + stitchImg_w - stitchImg_part.shape[1]
#         blank_image = np.zeros((height, width, 3), dtype=np.uint8)
#
#         blank_image[0:img_h, 0:img_w] = img1
#         blank_image[0:img_h, img_w - stitchImg_part.shape[1]:blank_image.shape[1]] = stitchPart
#         result = blank_image
#
#     if v == 'top':
#         templ = img1[0:img1.shape[0], 0:60]
#
#         result = cv2.matchTemplate(img2, templ, cv2.TM_CCORR_NORMED)
#         _, _, _, matchLoc = cv2.minMaxLoc(result)
#
#         top_left = matchLoc
#
#         stitchPart = img2[0:templ.shape[0], 0:top_left[0] + templ.shape[1]]
#
#         matchResult = img2[0:templ.shape[0], top_left[0]:templ.shape[1] + top_left[0]]
#
#         addImg = cv2.addWeighted(templ, 0.55, matchResult, 0.45, 0)
#         stitchPart[0:templ.shape[0], top_left[0]:templ.shape[1] + top_left[0]] = addImg
#
#         img_h, img_w = img1.shape[:2]
#         stitchImg_h, stitchImg_w = stitchPart.shape[:2]
#         height = img_h
#         width = img_w + stitchImg_w - templ.shape[1]
#         blank_image = np.zeros((height, width, 3), dtype=np.uint8)
#
#         blank_image[0:img_h, stitchImg_w - templ.shape[1]:blank_image.shape[1]] = img1
#         blank_image[0:img_h, 0:stitchImg_w] = stitchPart
#         result = blank_image
#
#     return result
