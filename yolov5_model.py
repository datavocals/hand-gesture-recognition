import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models/yolov5'))

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

class Yolov5Model:
    def __init__(self, 
            weights, 
            device='',
            imgsz=640,
            half=False,  # use FP16 half-precision inference
            ):
        # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=device, dnn=False) # dnn: use OpenCV DNN for ONNX inference
        stride, names, pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Half
        self.half &= (pt or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            self.model.model.half() if half else self.model.model.float()

        if pt and device.type != 'cpu':
            self.model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(self.model.model.parameters())))  # warmup

    def infer(self, image):
        # pre-process
        im = torch.from_numpy(image).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # inference
        return self.model(im)
        