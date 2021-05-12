import numpy as np
import time
from PIL import Image
import logging
from keras.applications.vgg16 import preprocess_input
from app.serving_client import BaseClient, AsyncClient, MaskRcnnClient
from app.config import MODEL_URL, DEBUG
from app.key_dicts import east_threshold, max_predict_size, ALPHABET
import app.utils as utils
from app.invoice_template.template import examples, tencent_name_transform
from app.extractor.information_extraction import DataHandle
from app.extractor.direction_filter_generator import get_direction_filter
import math
import os
import base64
import hmac
import app.crop_image as crop
import app.maskrcnn_detect as maskrcnn_detect
from app.maskrcnn_predict import unmold_detections
import app.maskrcnn_utils as rcnn_utils

# import matplotlib.pyplot as plt
import skimage.io
import app.maskrcnn_config as rcnn_config

## 获取模型地址
try:
    east_client = BaseClient(os.environ['MODEL_EAST'], 'east')
except KeyError:
    east_client = BaseClient(MODEL_URL['MODEL_EAST'], 'east')

try:
    ocr_client = AsyncClient(os.environ['MODEL_OCR'], 'ocr')
except KeyError:
    ocr_client = AsyncClient(MODEL_URL['MODEL_OCR'], 'ocr')

try:
    angle_client = BaseClient(os.environ['MODEL_ANGLE'], 'angle')
except KeyError:
    angle_client = BaseClient(MODEL_URL['MODEL_ANGLE'], 'angle')

try:
    maskrcnn_client = MaskRcnnClient(os.environ['MODEL_MASKRCNN'], 'maskrcnn')
except KeyError:
    maskrcnn_client = MaskRcnnClient(MODEL_URL['MODEL_MASKRCNN'], 'maskrcnn')

##
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

##
invoice_direction_filter = get_direction_filter(examples)


class OcrEngine(object):
    def __init__(self, east, ocr, angle, invoice_type=None, qrcode_dewarp=False, threshold=0.92, maskrcnn=None):
        self.invoice_type = invoice_type
        self.threshold = east_threshold.get(invoice_type, threshold)
        self.qrcode_dewarp = qrcode_dewarp
        self.east = east
        self.ocr = ocr
        self.angle = angle
        self.maskrcnn = maskrcnn
        self.box = []
        self.code = '200'
        self.msg = ''
        self.im, self.ocr_score, self.score_ = [], [], []

    def img_rotation(self, img):
        rotation_config = {0: None, 1: Image.ROTATE_270, 2: Image.ROTATE_180, 3: Image.ROTATE_90}
        angle = int(np.argmax(self.angle.predict(utils.read_for_rotation(img))[0]))
        # print("angle:",angle)

        return img.transpose(rotation_config[angle]) if rotation_config[angle] else img, angle

    def text_bbox(self, img):
        # img = np.where(np.array(img) < 240, 1, 255).astype('float32')
        img = np.array(img).astype('float32')
        img = preprocess_input(img, mode='tf')
        x = np.expand_dims(img, axis=0)
        y = self.east.predict(x)
        y = np.squeeze(y, axis=0)
        y[:, :, :3] = utils.sigmoid(y[:, :, :3])
        cond = np.greater_equal(y[:, :, 0], self.threshold)
        return utils.activate_box(cond)

    def read_img(self, img_path, img_base64, read_type="pil"):
        if img_path:
            try:
                img = utils.read_img_from_url(img_path, read_type)
                logger.info('img size: {}'.format(img.size))
            except OSError as e:
                self.code = '404'
                self.msg = 'Read img url {} error'.format(img_path)
                logger.error(self.msg)
                return [], [], [], []
        elif img_base64:
            try:
                img = utils.read_img_from_base64(img_base64, read_type)
            except OSError:
                self.code = '404'
                self.msg = 'Read img base64 error'
                logger.error(self.msg)
                return [], [], [], []
        else:
            self.code = '404'
            self.msg = 'Read img error'
            logger.error(self.msg)
            return [], [], [], []
        return img

    def img_2_text(self, img):
        img, angle = self.img_rotation(img)
        if self.qrcode_dewarp:
            img = utils.qrcode_dewarp(img)
        self.im = img

        d_wight, d_height = utils.resize_image(img, max_predict_size.get(self.invoice_type, 800))
        img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')

        im = self.im.convert('RGB')
        large = {}
        ocr_score, score_ = {}, {}

        def __predict(bb, mask, idx):
            try:
                x_gap = (bb[2] - bb[0]) / (img.width / im.width)
                y_gap = (bb[3] - bb[1]) / (img.height / im.height)
                bb_large = [math.ceil((bb[0] / (img.width / im.width)) - 1),
                            math.ceil((bb[1] / (img.height / im.height)) - 0.1 * y_gap),
                            math.ceil((bb[2] / (img.width / im.width)) + 1),
                            math.ceil((bb[3] / (img.height / im.height)) + 0.1 * y_gap)]
                large[idx] = bb_large
                corped_im = im.crop(bb_large)
                corped_im = utils.mask_croped_img(corped_im, mask, bb_large)
                self.ocr.predict(utils.handle_img(corped_im), __predict_callback, idx)
            except:
                ocr_score[idx] = ''
                score_[idx] = None
                self.code = '202'
                self.msg = 'img bbox ocr error'
                logger.error(self.msg)

        def __predict_callback(predict, i):
            ocr_score[i] = utils.decode_mask(predict)
            score_[i] = predict

        tasks = []
        for i, (_, box, mask) in enumerate(self.text_bbox(img)):
            tasks.append(i)
            __predict(box, mask, i)
        while len(ocr_score) < len(tasks):
            pass
        self.box = [i[1] for i in sorted(large.items(), key=lambda item: item[0])]
        self.ocr_score = [i[1][0] for i in sorted(ocr_score.items(), key=lambda item: item[0])]
        self.score_ = [i[1][0] for i in sorted(score_.items(), key=lambda item: item[0])]
        logger.info('ocr raw results: {}'.format(self.ocr_score))
        return self.ocr_score, self.box, self.score_, angle

    def predict(self, input_item):
        img_path, img_base64 = '', ''
        if input_item.ImageUrl:
            img_path = input_item.ImageUrl
        elif input_item.ImageBase64:
            img_base64 = input_item.ImageBase64
        ocr_score, box, score_, angle = self.img_2_text(self.read_img(img_path, img_base64))
        if self.code != '404':
            state, predict = DataHandle(ocr_score, box, score_, input_item.InvoiceType, invoice_direction_filter,
                                        True).extract()
        else:
            predict = []
            angle = -1
        return predict, self.code, angle, self.msg

    @utils.debug
    def segmentation(self, img_np):
        # print("img_np.shape:",img_np.shape)
        # print("img_np.shape:",type(img_np))

        # print("img_np[0]",img_np[0])

        # img_np = img_np.convert('RGB')

        img_np = img_np[:, :, 0:3]
        # print("img_np[1]", img_np[0])

        molded_images, image_metas, anchors, windows = maskrcnn_detect.detect([img_np])
        detections, mrcnn_mask = self.maskrcnn.predict(
            [molded_images.astype(np.float32), image_metas.astype(np.float32), anchors.astype(np.float32)])
        final_rois, final_class_ids, final_scores, final_masks = \
            unmold_detections(detections[0], mrcnn_mask[0], img_np.shape, molded_images[0].shape, windows[0])

        images_invoice = []
        angles_invoice = []
        re_list = []

        for i in range(final_masks.shape[-1]):
            rect = {"X": "",
                    "Y": "",
                    "Width": "",
                    "Height": ""
                    }
            re_dict = {"Angle": "",
                       "Type": "",
                       "Rect": rect,
                       "Image": None}

            try:
                image_invoice, angle_invoice = crop.get_cropimg_angle(img_np, final_masks[:, :, i])
            except:
                continue

            images_invoice.append(image_invoice)
            angles_invoice.append(angle_invoice)
            re_dict["Angle"] = angle_invoice

            class_id = rcnn_config.type_to_TengXun_TypeId[int(final_class_ids[i])]
            re_dict["Type"] = class_id

            y1, x1, y2, x2 = final_rois[i]
            rect["X"] = int(x1)
            rect["Y"] = int(y1)
            rect["Width"] = int(x2 - x1)
            rect["Height"] = int(y2 - y1)
            re_list.append(re_dict)

        return images_invoice, final_class_ids, re_list

    def segmentation_predict(self, input_item):
        img_path, img_base64 = '', ''
        if input_item.ImageUrl:
            img_path = input_item.ImageUrl
        elif input_item.ImageBase64:
            img_base64 = input_item.ImageBase64
        img = self.read_img(img_path, img_base64, read_type='skimage')
        predict = self.segmentation(img)
        return predict

    def multi_invoices_predict(self, input_item):
        img_path, img_base64 = '', ''
        if input_item.ImageUrl:
            img_path = input_item.ImageUrl
        elif input_item.ImageBase64:
            img_base64 = input_item.ImageBase64
        img = self.read_img(img_path, img_base64, read_type='skimage')
        pre_result = self.segmentation(img)

        # apply ocr for each img_nparray,
        images_invoice, class_ids, mask_angle = pre_result[0], pre_result[1], pre_result[2]
        # print("mask_angle:", mask_angle) # mask_angle是个list 要放在循环里
        # print("masks:", type(masks)) # mask_angle是个list 要放在循环里

        predicts = []
        angles = []
        # print("len(class_ids): ", len(class_ids))
        for i in range(len(class_ids)):
            images_invoice_shape = images_invoice[i].shape
            if images_invoice_shape[0] * images_invoice_shape[1] == 0:
                continue
            pil_im2 = Image.fromarray(np.uint8(images_invoice[i]))
            ocr_score, box, score_, angle = self.img_2_text(pil_im2)  # 需要图片格式转换

            # print("mask_angle[i]]:",mask_angle[i]["Angle"])
            # print("angle[i]]:",angle)

            angle = abs(mask_angle[i]["Angle"]) / 90 + angle  # mask_angle["Angle"][i]  (-90,0]
            angles.append(angle)

            if self.code != '404':
                # class_to_invoice_type = {7:"train_ticket"}
                # state, predict = DataHandle(ocr_score, box, score_, class_to_invoice_type[class_ids[i]], invoice_direction_filter, True).extract()

                # print("class_ids[i]:",class_ids[i])

                invoice_type = rcnn_config.class_to_invoice_type[class_ids[i]]

                print("invoice_type", invoice_type)

                if "bug" in invoice_type:
                    predict = "invoice_bug"
                    predicts.append(predict)
                else:
                    state, predict = DataHandle(ocr_score, box, score_, invoice_type, invoice_direction_filter,
                                                True).extract()

                    # print("predict.type: ",type(predict))
                    # print("predict.type: ", predict)
                    predicts.append(predict)
            else:
                angle = -1
                predicts.append([])
                angles.append(angle)
        return predicts, self.code, angles, self.msg


class OutputParser(object):

    def __init__(self, inputs, predicts, code, angle, msg):
        self.inputs = inputs
        self.predicts = predicts
        self.code = code
        self.angle = angle
        self.msg = msg
        self.response = {}

    def parse_output(self, invoice_type):
        self.response['pcId'] = self.inputs['pcId']

        if invoice_type == "dedicated_invoice":
            fp_list = [
                {"kKprq": utils.remove_punctuation(self.predicts[idx]['date']),
                 "kFphm": self.predicts[idx]['ticket_id'],
                 "kJehj": self.predicts[idx]['money'], "kFpdm": self.predicts[idx]['ticket_code'], "kFpid": fp['kFpid'],
                 "kFpdz": fp['kFpdz'], "code": self.predicts[idx]['code'], "angle": self.predicts[idx]['angle']} for
                idx, fp in enumerate(self.inputs['fpList'])]
            self.response['fpList'] = fp_list
            return self.response
        else:
            self.response['fpList'] = self.predicts
            return self.response


class TxOutputParser(OutputParser):
    def _decode(self, expire=36000):
        ts_str = str(time.time() + expire)
        ts_byte = ts_str.encode("utf-8")
        sha1_str = hmac.new(self.code.encode("utf-8"), ts_byte, 'sha1').hexdigest()
        secret = ts_str + ':' + sha1_str
        b64_secret = base64.urlsafe_b64encode(secret.encode("utf-8")).decode("utf-8")
        return b64_secret

    def parse_output(self, invoice_type):
        if invoice_type in tencent_name_transform:
            self.response["InvoiceInfos"] = {tencent_name_transform[invoice_type][k]: v for k, v in
                                             self.predicts.items() if k in tencent_name_transform[invoice_type]}
        else:
            self.response["InvoiceInfos"] = self.predicts
        self.response['RequestId'] = self._decode()
        self.response['Code'] = self.code
        self.response['Message'] = 'Success' if self.code == '200' else self.msg
        self.response['Angle'] = self.angle * 90
        return self.response


# [{"Angle": 89.52734375, "Type": 2, "Rect": {"X": 112, "Y": 0, "Width": 385, "Height": 571},
# "Image": null}, {"Angle": 90.6875, "Type": 2, "Rect": {"X": 439, "Y": 11, "Width": 396, "Height": 544}, "Image": null}, {"Angle": 88.953125, "Type": 2, "Rect": {"X": 423, "Y": 471, "Width": 437, "Height": 742}, "Image": null}, {"Angle": 90.48828125, "Type": 2, "Rect": {"X": 112, "Y": 498, "Width": 389, "Height": 765}, "Image": null}]

class SegmentationOutputParser(object):
    def __init__(self, inputs, predicts):
        self.inputs = inputs
        self.predict = predicts
        self.response = {}

    def parse_output(self):
        # print("self.predict:",self.predict)

        self.response['ocr_data'] = self.predict
        return self.response


class MultiOutputParser(object):

    def __init__(self, inputs, predicts, codes, angles, msgs):
        # print("print(len(predicts)):",len(predicts))
        self.inputs = inputs
        self.predicts = predicts
        self.code = codes
        self.angle = angles
        self.msg = msgs
        self.response = []

    def _decode(self, expire=36000):
        ts_str = str(time.time() + expire)
        ts_byte = ts_str.encode("utf-8")
        sha1_str = hmac.new(self.code.encode("utf-8"), ts_byte, 'sha1').hexdigest()
        secret = ts_str + ':' + sha1_str
        b64_secret = base64.urlsafe_b64encode(secret.encode("utf-8")).decode("utf-8")
        return b64_secret

    def parse_output(self):
        for i in range(len(self.predicts)):
            self.fp_dict = {}
            self.fp_dict["InvoiceInfos"] = self.predicts[i]
            self.fp_dict['RequestId'] = self._decode()
            self.fp_dict['Code'] = self.code
            self.fp_dict['Message'] = 'Success' if self.code == '200' else self.msg
            self.fp_dict['Angle'] = self.angle[i] * 90
            self.response.append(self.fp_dict)
        # print("self.response: ",self.response)

        return self.response
