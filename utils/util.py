import os
import pathlib

import cv2
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from numpy import sin, cos

pathlib.PosixPath = pathlib.WindowsPath


def check_save_dir(save_dir):
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)


def create_progress_bar(video_capture):
    total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = tqdm.tqdm(total=total)
    return progress


def create_video_write(video_capture, save_dir):
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    frame_width, frame_height = int(video_capture.get(3)), int(video_capture.get(4))
    output_path = pathlib.Path(save_dir)
    writer = cv2.VideoWriter(output_path.as_posix(), fourcc, fps - 2, (frame_width, frame_height))
    return writer


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=150, line_width=3):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx is not None and tdy is not None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    img = img.astype(np.uint8)
    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), line_width)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), line_width)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), line_width)

    return img
