import cv2
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'model/yolov5'))
from yolov5_model import Yolov5Model
from capture_frame_in_csi_camera import gstreamer_pipeline

model = Yolov5Model("model/weights/best.pt")

cap = cv2.VideoCapture(2)
while True:
    if cap.isOpened():
        retVal, img = cap.read()
        if retVal:
            img = cv2.resize(img, (640, 480))
            pred = model.infer(img)
            if pred != None:
                print(pred)
        else:
            continue
    else:
        cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)


