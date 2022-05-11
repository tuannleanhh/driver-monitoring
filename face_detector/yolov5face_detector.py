import cv2
import torch

from face_detector.models import YoloV5
from face_detector.utils import letterbox, check_img_size, non_max_suppression_face, scale_coords, xyxy2xywh

import matplotlib.pyplot as plt

class FaceDetector:

    def __init__(self, model_path, conf_thres=0.3, iou_thres=0.5):
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.model = self._load_model(model_path).to(self.device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = 320

    @staticmethod
    def _load_model(model_path):
        model = YoloV5(in_channels=3)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.eval()
        return model

    def _preprocess(self, data):
        orgimg = data
        h0, w0 = orgimg.shape[:2]
        img = orgimg.copy()
        r = self.img_size / max(w0, h0)

        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        imgsz = check_img_size(img_size=self.img_size, s=self.model.stride.max())
        img = letterbox(img, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()
        img = torch.from_numpy(img).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img, orgimg

    def _inference(self, img):
        img = img.to(self.device)
        with torch.no_grad():
            pred = self.model(img)
        return pred

    def _postprocess(self, pred, img, orgimg):
        pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)
        boxes = []
        scores = []
        h, w = orgimg.shape[:2]
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(self.device)  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

                for j in range(det.size()[0]):
                    xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                    conf = det[j, 4].detach().cpu().numpy()
                    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
                    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
                    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
                    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
                    boxes.append([x1, y1, x2, y2])
                    scores.append(conf)
        # box = [boxes[i] for i in range(len(boxes)) if scores[i] == max(scores)][0]
        box = boxes[0] if len(boxes) != 0 else [0, 0, w, h]
        return box

    def detect(self, data):
        img, orgimg = self._preprocess(data)
        pred = self._inference(img)
        box = self._postprocess(pred, img, orgimg)
        return box
