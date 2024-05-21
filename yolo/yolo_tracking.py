import random

import numpy as np
from ultralytics import YOLO
import cv2

from ultralytics.trackers.track import BYTETracker

from torch import tensor

source = "/home/an_nemenko/Videos/WIN_20240514_11_31_21_Pro.mp4"


class TrackerSettings:
    def __init__(self):
        self.tracker_type = "botsort"
        self.track_high_thresh = 0.5
        self.track_low_thresh = 0.1
        self.new_track_thresh = 0.6
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.gmc_method = "sparseOptFlow"
        self.proximity_thresh = 0.5
        self.appearance_thresh = 0.25
        self.with_reid = False


class BBOX:
    def __init__(self, _bbox):
        self.xywh = _bbox[0].boxes.xywh.to("cpu")
        self.conf = _bbox[0].boxes.conf.to("cpu")
        self.cls = _bbox[0].boxes.cls.to("cpu")


trc = BYTETracker(TrackerSettings())

model = YOLO('yolov8x.pt')  # load an official model
# model.track(source=source, show=True, conf=0.45, tracker="botsort.yaml")

cap = cv2.VideoCapture(source)

_r, frame = cap.read()
trc.reset()

while True:
    res = model.predict(frame)

    bb = BBOX(res)
    res = trc.update(bb, frame)

    print(res)

    if len(res) == 0:
        continue

    for i, bbox in enumerate(res):
        x, y, w, h = np.array(bbox[:4], dtype=int)
        ID, p, cls, id = bbox[4:]

        # if id != 1:
        #     continue

        color = (0, 0, 0)

        cv2.rectangle(frame, (x, y), (w, h), color=color, thickness=1)
        cv2.putText(img=frame, text=f"p:{p:.2f}; id:{ID}", org=(x, y - 10), color=color,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow(" ", frame)
    cv2.waitKey(20)

    _r, frame = cap.read()
