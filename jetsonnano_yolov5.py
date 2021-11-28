from yolov5_model import Yolov5Model
from utils.capture_frame_in_csi_camera import capture_frame

model = Yolov5Model("model/weights/best.pt")
pred = model.infer(capture_frame())
print(pred)
