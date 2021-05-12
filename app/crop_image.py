import numpy as np
import cv2
import math


# reference  https://www.pythonf.cn/read/73973
# def crop_image(image, box,padding=0):
#     xs = [x[1] for x in box]
#     ys = [x[0] for x in box]
#
#     # h, w, c = image.shape
#
#     min_xs = int(np.maximum(min(xs)-padding,0))
#     min_ys = int(np.maximum(min(ys)-padding,0))
#     max_xs = int(np.minimum(max(xs)+padding,image.shape[0]))
#     max_ys = int(np.minimum(max(xs)+padding,image.shape[0]))
#
#     # print("min(xs):",int(min(xs)), " max(xs):",int(max(xs)), "min(ys):",int(min(ys)), " max(ys):",int(max(ys)))
#
#     if int(max(xs))-int(min(xs))<10 or int(max(ys))-int(min(ys))<10: return image
#
#     # cropimage = image[int(min(xs)):int(max(xs)), int(min(ys)):int(max(ys))]
#     cropimage = image[min_xs:max_xs,min_ys:max_ys]
#
#     return cropimage


def crop_image(image, box):
    xs = [x[1] for x in box]
    ys = [x[0] for x in box]
    xs = np.maximum(xs, 0)
    ys = np.maximum(ys, 0)
    # print("min(xs):",int(min(xs)), " max(xs):",int(max(xs)), "min(ys):",int(min(ys)), " max(ys):",int(max(ys)))

    if int(max(xs)) - int(min(xs)) < 10 or int(max(ys)) - int(min(ys)) < 10: return image
    cropimage = image[int(min(xs)):int(max(xs)), int(min(ys)):int(max(ys))]

    return cropimage


# 逆时针旋转
def Nrotate(angle, valuex, valuey, pointx, pointy):
    angle = (angle / 180) * math.pi
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    nRotatex = (valuex - pointx) * math.cos(angle) - (valuey - pointy) * math.sin(angle) + pointx
    nRotatey = (valuex - pointx) * math.sin(angle) + (valuey - pointy) * math.cos(angle) + pointy
    return (nRotatex, nRotatey)


# 顺时针旋转
def Srotate(angle, valuex, valuey, pointx, pointy):
    angle = (angle / 180) * math.pi
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    sRotatex = (valuex - pointx) * math.cos(angle) + (valuey - pointy) * math.sin(angle) + pointx

    # print("sRotatex:",sRotatex)
    sRotatey = (valuey - pointy) * math.cos(angle) - (valuex - pointx) * math.sin(angle) + pointy
    # print("sRotatey:", sRotatey)
    return (sRotatex, sRotatey)

    # 将四个点做映射


def rotatecordiate(angle, rectboxs, pointx, pointy):
    output = []
    for rectbox in rectboxs:
        if angle > 0:
            output.append(Srotate(angle, rectbox[0], rectbox[1], pointx, pointy))
        else:
            output.append(Nrotate(-angle, rectbox[0], rectbox[1], pointx, pointy))
    return output


def get_cropimg_angle(image, image_mask):
    image_mask = np.where(image_mask > 0.5, 255, 0)
    image_mask = image_mask.astype(np.uint8)

    print("image_mask:", image_mask.shape)

    # gray_mask = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(image_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours, find rotated rectangle, obtain four verticies, and draw
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    rect = cv2.minAreaRect(cnts[0])
    # try:
    #     rect = cv2.minAreaRect(cnts[0])
    # except:
    #     continue

    # print("angle:", rect[2])
    box_origin = np.int0(cv2.boxPoints(rect))
    M = cv2.getRotationMatrix2D(rect[0], rect[2], 1)
    # dst = cv2.warpAffine(image_mask, M, (2 * image_mask.shape[0], 2 * image_mask.shape[1]))
    # dst_mask = cv2.warpAffine(image_mask, M, (2 * image_mask.shape[0], 2 * image_mask.shape[1]))
    dst_image = cv2.warpAffine(image, M, (2 * image.shape[0], 2 * image.shape[1]))

    box = rotatecordiate(rect[2], box_origin, rect[0][0], rect[0][1])

    crop_img = crop_image(dst_image, box)
    mask_angle = rect[2]

    return crop_img, mask_angle

    # cv2.imwrite('/home/yzf/data2/{}.png'.format(str(i)), crop_image)
