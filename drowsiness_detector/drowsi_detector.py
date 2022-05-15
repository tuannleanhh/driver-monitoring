import time

import cv2
import mediapipe
import numpy as np
import onnxruntime


class DrowsinessDetector:
    REYE_INDICES = np.array([33, 133])
    LEYE_INDICES = np.array([362, 263])

    def __init__(self, model_path):
        self.model = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.landmark_detector = mediapipe.solutions.face_mesh.FaceMesh()

    @staticmethod
    def _preprocess_img(img):
        try:
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            img = np.expand_dims(img, axis=0).astype(np.float32)
            img = img.transpose((0, 3, 1, 2))
            return img
        except Exception as e:
            return None

    @staticmethod
    def padding(eye):
        eye[0][0] -= 0.3 * (eye[1][0] - eye[0][0])
        eye[0][1] -= 0.7 * (eye[1][0] - eye[0][0])
        eye[1][0] += 0.3 * (eye[1][1] - eye[0][1])
        eye[1][1] += 0.6 * (eye[1][1] - eye[0][1])
        return eye

    @staticmethod
    def crop_eye(img, x_min, y_min, x_max, y_max):
        k = np.random.random_sample() * 0.15
        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += k * abs(y_max - y_min)
        return img[int(y_min): int(y_max), int(x_min): int(x_max)]

    def _detect_eyes(self, img):
        h, w = img.shape[:2]
        predictions = self.landmark_detector.process(img[:, :, ::-1])
        landmarks = predictions.multi_face_landmarks

        if landmarks is not None:
            prediction = landmarks[0]
            pts = np.array([(pt.x * w, pt.y * h) for pt in prediction.landmark], dtype=np.float64)
            reye = self.padding(pts[self.REYE_INDICES])
            leye = self.padding(pts[self.LEYE_INDICES])
            reye_img = self.crop_eye(img.copy(), reye[0][0], reye[0][1], reye[1][0], reye[1][1])
            leye_img = self.crop_eye(img.copy(), leye[0][0], leye[0][1], leye[1][0], leye[1][1])
            return reye_img, leye_img
        else:
            return None, None

    def inference(self, image):
        ort_input = {self.model.get_inputs()[0].name: image}
        pred = self.model.run(None, ort_input)
        pred = np.argmax(pred[0], axis=1)
        return pred

    def detect(self, image):
        reye_image, leye_image = self._detect_eyes(image.copy())
        if reye_image is not None and leye_image is not None:
            reye_image, leye_image = self._preprocess_img(reye_image), self._preprocess_img(leye_image)
            if reye_image is not None and leye_image is not None:
                eyes_image = np.concatenate((reye_image, leye_image), axis=0)
                reye_pred, leye_pred = self.inference(eyes_image)
                return reye_pred, leye_pred
            else:
                return None, None
        else:
            return None, None
