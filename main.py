import time

from drowsiness_detector import DrowsinessDetector
from face_detector import FaceDetector
from head_pose_estimator import HopeNetEstimator
from utils import *


def run_on_video(video_path, save_dir, plot=False, save=False):
    face_detector = FaceDetector('checkpoints/torch_ckpts/yolo-ep299.pt')
    drowsiness_detector = DrowsinessDetector('checkpoints/tflite_ckpts/drowsiness.tflite')
    head_pose_estimator = HopeNetEstimator('checkpoints/onnx_ckpts/hopenet.onnx')
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # video_capture = cv2.VideoCapture(video_path)
    check_save_dir(save_dir)
    writer = create_video_write(video_capture, save_dir)
    progressbar = create_progress_bar(video_capture)

    count_head_pose = 0
    count_drowsiness = 0
    sum_frame = 0
    fps = 0
    while True:
        ok, im = video_capture.read()
        if not ok:
            break

        image = im.copy()
        t1 = time.time()
        img = image.copy()

        # detect face and draw face box
        x_min, y_min, x_max, y_max = face_detector.detect(img)
        face = img.copy()[int(y_min): int(y_max), int(x_min): int(x_max)]
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

        # Detect head pose and draw pose + type face
        yaw, pitch, roll, type_face = head_pose_estimator.estimate(img, x_min, y_min, x_max, y_max)
        cv2.putText(img, f"yaw: {yaw:.2f}", (int(img.shape[1] * 0.03), int(img.shape[0] * 0.04)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (52, 236, 76), 2, cv2.LINE_AA)
        cv2.putText(img, f"pitch: {pitch:.2f}", (int(img.shape[1] * 0.03), int(img.shape[0] * 0.08)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (52, 236, 76), 2, cv2.LINE_AA)
        cv2.putText(img, f"roll: {roll:.2f}", (int(img.shape[1] * 0.03), int(img.shape[0] * 0.12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (52, 236, 76), 2, cv2.LINE_AA)
        cv2.putText(img, type_face, (int(x_min), int(y_min - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (52, 236, 76), 2, cv2.LINE_AA)
        img = draw_axis(img, yaw, pitch, roll, tdx=(x_max - x_min) / 2 + x_min,
                        tdy=(y_max - y_min) / 2 + y_min, size=150)

        # draw head pose alert
        if pitch is not None and yaw is not None:
            if pitch < -15 or pitch > 15 or yaw < -12 or yaw > 12:
                count_head_pose += 1
            else:
                count_head_pose = 0
            if count_head_pose >= 15:
                cv2.putText(img, "ALERT!!!", (int(image.shape[1] * 0.1), int(image.shape[0] * 0.45)),
                            cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 2,
                            cv2.LINE_AA)

        # draw drowsiness alert
        if type_face == 'no mask' and count_head_pose == 0:
            reye, leye = drowsiness_detector.detect(face.copy())
            if reye == 0 or leye == 0:
                count_drowsiness += 1
                if count_drowsiness >= 30:
                    cv2.putText(img, "DROWSINESS ALERT!!!", (int(image.shape[1] * 0.02), int(image.shape[0] * 0.45)),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2,
                                cv2.LINE_AA)
            elif reye == 1 and leye == 1:
                count_drowsiness = 0

        t2 = time.time()
        cv2.putText(img, f"fps: {(1 / (t2 - t1)):.2f}", (int(img.shape[1] * 0.8), int(img.shape[0] * 0.04)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (52, 236, 76), 2, cv2.LINE_AA)
        if save:
            writer.write(img)
        if plot:
            cv2.imshow('frame', img)
            if cv2.waitKey(1) == ord('q'):
                break

        fps += (1 / (t2 - t1))
        sum_frame += 1
        progressbar.update(1)

    print(f'\nAVERAGE THREAD FPS: {fps / sum_frame}')
    video_capture.release()
    cv2.destroyAllWindows()
