import torch
import cv2
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
        self.stride, names, pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        # Half
        half &= (pt or engine) and self.device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            self.model.model.half() if half else self.model.model.float()
        self.half = half

        if pt and self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *imgsz).to(self.device).type_as(next(self.model.model.parameters())))  # warmup

    def infer(self, image):
        image = cv2.resize(image, (640, 480))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.moveaxis(image, -1 ,0)

        # pre-process
        im = torch.from_numpy(image).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)
        
        # inference
        pred = self.model(im)

        # NMS
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

        items = []
        
        if pred[0] is not None and len(pred):
            for p in pred[0]:
                score = np.round(p[4].cpu().detach().numpy(),2)
                label = int(p[5])

                item = {'label': label,
                        'score': score
                        }

                items.append(item)

        return items
        