import torch
import numpy as np
from model.yolov5.models.common import DetectMultiBackend
from model.yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from model.yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from model.yolov5.utils.plots import Annotator, colors, save_one_box
from model.yolov5.utils.torch_utils import select_device, time_sync
from model.yolov5.utils.augmentations import letterbox

class Yolov5Model:
    def __init__(self, 
            weights, 
            device='',
            imgsz=[640, 640],
            half=False,  # use FP16 half-precision inference
            ):
        # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device , dnn=False) # dnn: use OpenCV DNN for ONNX inference
        self.stride, self.names, pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        # Half
        half &= (pt or engine) and self.device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            self.model.model.half() if half else self.model.model.float()
        self.half = half

        if pt and self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *imgsz).to(self.device).type_as(next(self.model.model.parameters())))  # warmup

    def infer(self, image):
        # change img data formation
        image = letterbox(image, self.imgsz, stride=self.stride, auto=True)[0]
        image = image.transpose((2, 0, 1))[::-1] # HWC to CHW, BGR to RGB
        image = np.ascontiguousarray(image)

        # pre-process
        im = torch.from_numpy(image).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # inference
        pred = self.model(im, augment=False, visualize=False)
        # NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None)
        
        if len(pred[0]) > 0:
            return self.names[int(pred[0][0][5])]
        else:
            return None
        