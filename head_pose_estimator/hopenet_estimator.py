import time

import cv2
import numpy as np
import onnxruntime


class HopeNetEstimator:

    def __init__(self, checkpoint_path):
        self.model = onnxruntime.InferenceSession(checkpoint_path, providers=['CPUExecutionProvider'])

    @staticmethod
    def _preprocess(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img = img.transpose((0, 3, 1, 2))
        return img

    @staticmethod
    def crop_face(img, x_min, y_min, x_max, y_max):
        k = 0.2
        face_pad_w = k * (x_max - x_min)
        face_pad_h = k * (y_max - y_min)
        x_min = x_min - face_pad_w
        y_min = y_min - (face_pad_h + 0.2 * face_pad_h)
        x_max = x_max + face_pad_w
        y_max = y_max + (face_pad_h + 0.1 * face_pad_h)

        return img[int(y_min): int(y_max), int(x_min): int(x_max)]

    def inference(self, image):
        ort_input = {self.model.get_inputs()[0].name: image}
        t1 = time.time()
        pred = self.model.run(None, ort_input)
        t2 = time.time()
        yaw, pitch, roll, type_face = pred

        return yaw, pitch, roll, type_face, 1 / (t2 - t1)

    def estimate(self, image, x_min, y_min, x_max, y_max):
        image = self.crop_face(image.copy(), x_min, y_min, x_max, y_max)
        if image is not None and image.shape[0] > 0 and image.shape[1] > 0:
            face_img = self._preprocess(image.copy())
            yaw, pitch, roll, type_face, fps = self.inference(face_img)
            type_face = 'mask' if np.argmax(type_face, axis=1) else 'no mask'
            return yaw.item(), pitch.item(), roll.item(), type_face
        else:
            return None, None, None, None
