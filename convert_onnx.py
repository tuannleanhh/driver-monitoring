import pathlib
import time

import cv2
import numpy as np
import onnx
import onnxruntime as onnxrt
import torch

from face_detector.models import YoloV5

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_torch_model():
    pathlib.PosixPath = pathlib.WindowsPath
    model = YoloV5()
    checkpoint = "checkpoints/torch_ckpts/yolo-ep299.pt"
    state_dict = torch.load(checkpoint)['model']
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def convert():
    model = load_torch_model()
    batch_size = 1
    x = torch.rand(batch_size, 3, 320, 320, requires_grad=True).to(device)
    dynamic = False

    torch.onnx.export(model,
                      x,
                      "checkpoints/onnx_ckpts/yolov5face.onnx",
                      verbose=False,
                      export_params=True,
                      opset_version=12,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      input_names=["input"],
                      output_names=["output"],
                      dynamic_axes={'input':
                                        {0: 'batch_size',
                                         2: 'height',
                                         3: 'width'},
                                    'output':
                                        {0: 'batch_size',
                                         1: 'anchors'}} if dynamic else None
                      )

    onnx_model = onnx.load("checkpoints/onnx_ckpts/yolov5face.onnx")
    onnx.checker.check_model(onnx_model)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def _preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 192))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    img = torch.from_numpy(img).permute(0, 3, 1, 2)
    return img


def inference_onnx():
    batch_size = 1
    img = cv2.imread('testing_media/testing_images/image_12.jpg')
    x = _preprocess(img).to(device)

    model = load_torch_model()
    t1 = time.time()
    torch_out = model(x)
    print(f'torch fps: {1 / (time.time() - t1)}')
    print(torch_out)

    onnx_session = onnxrt.InferenceSession('checkpoints/onnx_ckpts/yolov5face.onnx',
                                           providers=['CUDAExecutionProvider'])
    ort_inputs = {onnx_session.get_inputs()[0].name: to_numpy(x)}
    t2 = time.time()
    ort_outs = onnx_session.run(None, ort_inputs)
    print(f'onnx fps: {1 / (time.time() - t2)}')

    print(onnxrt.get_device())
    print(ort_outs)


if __name__ == '__main__':
    convert()

    # from face_detector import FaceDetector
    # face_detector = FaceDetector('checkpoints/torch_ckpts/yolo-ep299.pt')
    # img = cv2.imread('testing_media/testing_images/image_12.jpg')
    # img = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))
    # x_min, y_min, x_max, y_max = face_detector.detect(img)
    # cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
