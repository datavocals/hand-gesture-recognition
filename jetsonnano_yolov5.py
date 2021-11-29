import cv2
import os
import sys
import subprocess
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'model/yolov5'))
from yolov5_model import Yolov5Model
from capture_frame_in_csi_camera import gstreamer_pipeline

# set speed when communicate with arduino
subprocess.Popen(["stty", "9600", "-F",  "/dev/ttyACM0", "raw", "-echo"]) # arduino device id on ubuntu: /dev/ttyACM0
dev = os.open("/dev/ttyACM0", os.O_RDWR)


model = Yolov5Model("model/weights/best.pt")

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

while True:
    if cap.isOpened():
        retVal, img = cap.read()
        if retVal:
            img = cv2.resize(img, (640, 480))
            pred = model.infer(img)
            if pred != None:
                os.write(dev, (pred + "\n").encode())
                print("sending %s" % pred)
        else:
            continue
    else:
        cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)


