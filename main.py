from ultralytics import YOLO
from utils import speed_estimation
import cv2
import numpy as np

# choose model
# model = YOLO("model/yolov8n.pt")
model = YOLO("model/yolov8m.pt")
names = model.model.names

# load video
cap = cv2.VideoCapture("video/video_demo.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

# Video writer
video_writer = cv2.VideoWriter(
    "results/speed_estimation.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

# Init speed-estimation obj
# you need to complete it yourself base you camera view
SOURCE = np.array([[268, 240], [515, 240], [487, 375], [115, 375]])

TARGET_WIDTH = 7
TARGET_HEIGHT = 15

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

speed_obj = speed_estimation.SpeedEstimator()
speed_obj.set_args(
    source_region=SOURCE, target_region=TARGET, fps=fps, names=names, view_img=True
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print(
            "Video frame is empty or video processing has been successfully completed."
        )
        break

    # or you can choose other tracker
    tracks = model.track(im0, persist=True, show=False, tracker="bytetrack.yaml")
    im0 = speed_obj.estimate_speed(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
