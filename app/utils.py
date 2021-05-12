import numpy as np
import requests as req
from io import BytesIO
import base64
from PIL import Image
import cv2
import math
import sys
import re
import itertools
from pyzbar.pyzbar import decode
from scipy.ndimage import affine_transform
from keras.preprocessing.image import img_to_array
from keras import backend as K
import skimage
import skimage.io
from skimage.measure import regionprops
from skimage.morphology import label
import time

from app.key_dicts import ALPHABET

alphabet = ALPHABET + u'卍'


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def activate_box(cond):
    label_img = label(cond)
    props = regionprops(label_img)
    for i in props:
        cord = i['BoundingBox']
        mask = i.convex_image
        x_gap = cord[3] - cord[1]
        y_gap = cord[2] - cord[0]
        if ((x_gap > 2) or (y_gap > 2)) and y_gap / x_gap < 4:
            box = [(cord[0], cord[1]), (cord[2], cord[1]), (cord[2], cord[3]), (cord[0], cord[3])]
            box_m = [max(math.ceil((cord[1] - 1) * 4), 0),
                     max(math.ceil((cord[0] - 1) * 4), 0),
                     max(math.ceil((cord[3] + 1) * 4), 0),
                     max(math.ceil((cord[2] + 1) * 4), 0)]
            yield box, box_m, mask


def handle_img(im):
    img = im.convert('RGB')
    scale = img.size[1] * 1.0 / 32
    w = int(img.size[0] / scale)
    img = img.resize((w, 32), Image.BILINEAR)
    img_array = np.asarray(img) / 255
    return np.expand_dims(np.transpose(img_array, (1, 0, 2)).astype('float32'), 0)


def mask_croped_img(croped_im, mask, crop_box):
    bb_size = (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])
    mask = np.transpose(mask)
    resized_mask = skimage.transform.resize(mask, bb_size) * 255
    resized_mask = Image.fromarray(np.transpose(np.uint8(resized_mask)))
    im_size = croped_im.size
    flat_im = np.reshape(np.array(croped_im), (im_size[0] * im_size[1], 3))
    H, edges = np.histogramdd(flat_im, bins=(8, 8, 8))
    color_bin_idx = np.unravel_index(H.argmax(), H.shape)
    color = [int((edges[i][color_bin_idx[i] + 1] + edges[i][color_bin_idx[i]]) / 2) for i in range(3)]
    background = Image.new("RGB", bb_size, tuple(color))
    return Image.composite(croped_im, background, resized_mask)


def text_to_labels(text):
    ret = []
    for char in text:
        find_idx = alphabet.find(char)
        if find_idx != -1:
            ret.append(find_idx)
    return ret


def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)


def labels_to_text_(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("卍")
        else:
            ret.append(alphabet[c])
    return "".join(ret)


def decode_original(out):
    ret = []
    out_best = list(np.argmax(out, 1))
    out_str = labels_to_text_(out_best)
    ret.append(out_str)
    return ret


def decode_mask(out, mask=np.array([])):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:] * mask, 1)) if mask.any() else list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        out_str = labels_to_text(out_best)
        ret.append(out_str)
    return ret


def read_img_from_url(url, read_type="pil"):
    response = req.get(url)
    if read_type == "pil":
        return Image.open(BytesIO(response.content))
    elif read_type == "skimage":
        return skimage.io.imread(BytesIO(response.content))


def read_img_from_base64(base64_code, read_type="pil"):
    base64_code = base64_code.replace('<img src="', '')
    base64_code = base64_code.replace('" alt=""/>', '')
    base64_code = base64_code.replace("<img src='", '')
    base64_code = base64_code.replace("' alt=''/>", '')
    text = re.findall(r'data:image/.*;base64,', base64_code)
    if text:
        base64_code = base64_code.replace(text[0], '')

    if read_type == "pil":
        return Image.open(BytesIO(base64.b64decode(base64_code)))
    elif read_type == "skimage":
        return skimage.io.imread(BytesIO(base64.b64decode(base64_code)))


def resize_image(im, max_img_size):
    im_width = np.minimum(im.width, max_img_size)
    if im_width == max_img_size < im.width:
        im_height = int((im_width / im.width) * im.height)
    else:
        im_height = im.height
    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height


def remove_punctuation(line):
    rule = re.compile(u'[^0-9]')
    line = rule.sub('', line)
    return line


def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    rotation = np.deg2rad(rotation)
    shear = np.deg2rad(shear)
    rotation_matrix = np.array(
        [[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shear_matrix = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix = np.array([[1.0 / height_zoom, 0, 0], [0, 1.0 / width_zoom, 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))


def center_transform(affine, input_shape, img_shape):
    hi, wi = float(input_shape[0]), float(input_shape[1])
    ho, wo = float(img_shape[0]), float(img_shape[1])
    top, left, bottom, right = 0, 0, hi, wi
    ''' 对比例特殊的图像进行裁剪
    if wi/hi/anisotropy < wo/ho: # input image too narrow, extend width
        w     = hi*wo/ho*anisotropy
        left  = (wi-w)/2
        right = left + w
    else: # input image too wide, extend height
        h      = wi*ho/wo/anisotropy
        top    = (hi-h)/2
        bottom = top + h
    '''
    center_matrix = np.array([[1, 0, -ho / 2], [0, 1, -wo / 2], [0, 0, 1]])
    scale_matrix = np.array([[(bottom - top) / ho, 0, 0], [0, (right - left) / wo, 0], [0, 0, 1]])
    decenter_matrix = np.array([[1, 0, hi / 2], [0, 1, wi / 2], [0, 0, 1]])
    return np.dot(np.dot(decenter_matrix, scale_matrix), np.dot(affine, center_matrix))


def transform_img(x, affine, img_shape):
    matrix = affine[:2, :2]
    offset = affine[:2, 2]
    x = np.moveaxis(x, -1, 0)
    channels = [affine_transform(channel, matrix, offset, output_shape=img_shape[:-1], order=1,
                                 mode='constant', cval=np.average(channel)) for channel in x]
    return np.moveaxis(np.stack(channels, axis=0), 0, -1)


def read_for_rotation(img, img_shape=(224, 224, 3)):
    x = img_to_array(img.convert('RGB'))
    data = np.zeros((1,) + img_shape, dtype=K.floatx())
    t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t = center_transform(t, x.shape, img_shape)
    x = transform_img(x, t, img_shape)
    x -= np.mean(x, keepdims=True)
    x /= np.std(x, keepdims=True) + K.epsilon()
    data[0] = x
    return data


def _cal_center_point(points):
    x = [points[i][0] for i in range(4)]
    y = [points[i][1] for i in range(4)]
    center = [int(sum(x) / 4), int(sum(y) / 4)]
    left_up = [point for point in points if (point[0] < center[0]) and (point[1] < center[1])]
    left_down = [point for point in points if (point[0] < center[0]) and (point[1] > center[1])]
    right_down = [point for point in points if (point[0] > center[0]) and (point[1] > center[1])]
    right_up = [point for point in points if (point[0] > center[0]) and (point[1] < center[1])]
    return center, [left_up[0], left_down[0], right_down[0], right_up[0]]


def _cal_edge_length(points):
    distanse = [int(math.sqrt((points[i][0] - points[i + 1][0]) ** 2 + (points[i][1] - points[i + 1][1]) ** 2)) for i in
                range(3)]
    return min(distanse)


def qrcode_dewarp(pil_img):
    qrcode_result = decode(pil_img)
    if qrcode_result:
        points = [(i[0], i[1]) for i in qrcode_result[0][3]]
        center, points = _cal_center_point(points)
        length = _cal_edge_length(points)
        half = int(length * 0.4)
        dewarp_points = [(center[0] - half, center[1] - half),
                         (center[0] - half, center[1] + half),
                         (center[0] + half, center[1] + half),
                         (center[0] + half, center[1] - half)]
        M = cv2.getPerspectiveTransform(np.float32(points), np.float32(dewarp_points))
        cv2_img = np.asarray(pil_img)
        dst = cv2.warpPerspective(cv2_img, M, (cv2_img.shape[1], cv2_img.shape[0]))
        return Image.fromarray(dst)
    else:
        return pil_img


def debug(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            score = fn(*args, **kwargs)
            print("{} cost {}".format(fn.__name__, time.time() - start))
            return score
        except Exception as e:
            print("{} except {}".format(fn.__name__, repr(e)))

    return wrapper
