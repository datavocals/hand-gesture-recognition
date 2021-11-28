import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'model/yolov5'))
from yolov5_model import Yolov5Model
from capture_frame_in_csi_camera import capture_frame

model = Yolov5Model("model/weights/best.pt")
img = capture_frame()[1]
pred = model.infer(img)
print(pred)
