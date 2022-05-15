import cv2
import numpy as np
import onnxruntime
import torch

from face_detector.utils import letterbox, check_img_size, non_max_suppression_face, scale_coords, xyxy2xywh, \
    DecoderYoloV5


class FaceDetector:

    def __init__(self, model_path, conf_thresh=0.3, iou_thresh=0.5):
        self.model = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = 320
        self.decoder = DecoderYoloV5()

    def _preprocess(self, data):
        orgimg = data
        h0, w0 = orgimg.shape[:2]
        img = orgimg.copy()
        r = self.img_size / max(w0, h0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        imgsz = check_img_size(img_size=self.img_size, s=32)
        img = letterbox(img, new_shape=imgsz)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).copy().astype(np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        return img, orgimg

    def _inference(self, img):
        ort_input = {self.model.get_inputs()[0].name: img}
        pred = self.model.run(None, ort_input)[0]
        pred = self.decoder.forward(pred)

        return pred

    def _postprocess(self, pred, img, orgimg):
        pred = non_max_suppression_face(pred, self.conf_thresh, self.iou_thresh)
        boxes = []
        scores = []
        h, w = orgimg.shape[:2]
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
                for j in range(det.shape[0]):
                    xywh = (xyxy2xywh(det[j, :4].reshape(1, 4)) / gn).reshape(-1).tolist()
                    conf = det[j, 4]
                    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
                    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
                    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
                    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
                    boxes.append([x1, y1, x2, y2])
                    scores.append(conf)
        box = boxes[0] if len(boxes) != 0 else [0, 0, w, h]
        return box

    def detect(self, data):
        img, orgimg = self._preprocess(data)
        pred = self._inference(img)
        x_min, y_min, x_max, y_max = self._postprocess(pred, img, orgimg)
        return x_min, y_min, x_max, y_max
