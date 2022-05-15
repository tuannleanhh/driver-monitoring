from face_detector import FaceDetector
import cv2
import matplotlib.pyplot as plt


def main():
    face_detector = FaceDetector('checkpoints/onnx_ckpts/yolov5face.onnx')

    img = cv2.imread('testing_media/testing_images/image_11.jpg')
    x_min, y_min, x_max, y_max, _ = face_detector.detect(img)

    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 5)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
