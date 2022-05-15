import pathlib
import time

import cv2
import numpy as np
import onnx
import onnxruntime as onnxrt
import torch

from face_detector.models import YoloV5
from drowsiness_detector.model import MobileNetV2
from face_detector.utils import letterbox, check_img_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_torch_model():
    pathlib.PosixPath = pathlib.WindowsPath
    model = YoloV5()
    checkpoint = "checkpoints/torch_ckpts/yolo-ep299.pt"
    state_dict = torch.load(checkpoint)['model']
    # model = MobileNetV2()
    # state_dict = torch.load("checkpoints/torch_ckpts/checkpoint-epoch5.pth", map_location='cpu')['state_dict']
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
                      verbose=True,
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


# def _preprocess(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (320, 320))
#     img = img / 255.0
#     img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
#     img = np.expand_dims(img, axis=0).astype(np.float32)
#     img = torch.from_numpy(img).permute(0, 3, 1, 2)
#     return img

def _preprocess(data):
    orgimg = data
    h0, w0 = orgimg.shape[:2]
    img = orgimg.copy()
    r = 320 / max(w0, h0)

    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    imgsz = check_img_size(img_size=320, s=32)
    img = letterbox(img, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()
    img = torch.from_numpy(img).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img, orgimg


def inference_onnx():
    batch_size = 2
    img0 = cv2.imread('testing_media/testing_images/image_12.jpg')
    # img1 = cv2.imread('testing_media/testing_images/image_11.jpg')
    x, _ = _preprocess(img0)
    # x1 = _preprocess(img1)
    # x = torch.cat((x0, x1), dim=0).to(device)

    with torch.no_grad():
        model = load_torch_model()
        t1 = time.time()
        torch_out = model(x.to(device))
        print(f'>>> torch fps: {1 / (time.time() - t1)}')
        print(torch_out)
        print(torch_out.shape)

    onnx_session = onnxrt.InferenceSession('checkpoints/onnx_ckpts/yolov5face.onnx',
                                           providers=['CUDAExecutionProvider'])
    ort_inputs = {onnx_session.get_inputs()[0].name: to_numpy(x)}
    t2 = time.time()
    ort_outs = onnx_session.run(None, ort_inputs)
    print(f'>>> onnx fps: {1 / (time.time() - t2)}')

    print(onnxrt.get_device())
    print(ort_outs, ort_outs[0].shape)

    loss = torch.nn.L1Loss()
    print(loss(torch_out.cpu(), torch.from_numpy(ort_outs[0])))


if __name__ == '__main__':
    # convert()
    # inference_onnx()

    from face_detector import FaceDetector
    face_detector = FaceDetector('checkpoints/torch_ckpts/yolo-ep299.pt')
    img = cv2.imread('testing_media/testing_images/image_11.jpg')
    img = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))
    x_min, y_min, x_max, y_max, _ = face_detector.detect(img)
    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
