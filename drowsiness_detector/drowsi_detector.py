import cv2
import torch
import mediapipe
import numpy as np
import tensorflow as tf

import onnxruntime
from drowsiness_detector.model import MobileNetV2


class DrowsinessDetector:
    REYE_INDICES = np.array([33, 133])
    LEYE_INDICES = np.array([362, 263])

    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.model = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        self.model = self._load_model(model_path)#.to(self.device)
        self.landmark_detector = mediapipe.solutions.face_mesh.FaceMesh()

    @staticmethod
    def _load_model(model_path):
        # state_dict = torch.load(model_path)['state_dict']
        # model = MobileNetV2()
        # model.load_state_dict(state_dict)
        # model.eval()
        # return model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details

    @staticmethod
    def _preprocess_img(img):
        try:
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            img = np.expand_dims(img, axis=0).astype(np.float32)
            # img = torch.from_numpy(img).permute(0, 3, 1, 2)
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
        # pred = self.model(image)
        # pred = np.argmax(pred.cpu().detach().numpy(), axis=1)

        interpreter, input_details, output_details = self.model
        interpreter.set_tensor(input_details[0]["index"], image)
        interpreter.invoke()
        pred = np.argmax(np.array(interpreter.get_tensor(output_details[0]['index'])[0]), axis=0)
        return pred

    def detect(self, image):
        reye_image, leye_image = self._detect_eyes(image.copy())
        if reye_image is not None and leye_image is not None:
            reye_image, leye_image = self._preprocess_img(reye_image), self._preprocess_img(leye_image)
            if reye_image is not None and leye_image is not None:
                reye_pred = self.inference(reye_image)#.to(self.device))
                leye_pred = self.inference(leye_image)#.to(self.device))
                return reye_pred, leye_pred
            else:
                return None, None
        else:
            return None, None
