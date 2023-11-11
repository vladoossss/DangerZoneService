import argparse
from collections import defaultdict
import os
import platform
import sys
from pathlib import Path
from typing import Any
from time import time
import pandas as pd
from tqdm import tqdm

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox
from glob import glob
import numpy as np
import re
from shapely.geometry import Point, Polygon


def init_danger_zone(path2fol):
    danger_zone_polies = defaultdict(list)

    camera_names = [i.replace('danger_', "").replace('.txt', '') for i in os.listdir(path2fol)]
    for camera_name in camera_names:
        txt_path = os.path.join(path2fol, f'danger_{camera_name}.txt')
        with open(txt_path) as f:
            poly = np.array([int(re.sub("[^0-9]", "",i)) for i in f.read().split(",")]).reshape(-1, 2)
            if 'zone' in txt_path:
                camera_name = camera_name.split('_zone')[0]
            danger_zone_polies[camera_name].append(poly)

    return danger_zone_polies


class YoloPredictor:
    def __init__(self,
            weights=ROOT / 'yolov5s.pt',  # model path or triton URL
            data='data/people.yaml',  # dataset.yaml path
            imgsz=(1024, 1024),  # inference size (height, width)
            conf_thres=0.6,  # confidence threshold
            iou_thres=0.5,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            nosave=False,  # do not save images/videos
            classes=[0],  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            update=False,  # update all models
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
        ) -> None:

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Load danger zone info
        danger_zone_poly = init_danger_zone(path2fol='cameras_data/danger_zones')

        self.__dict__.update(locals())

    def get_dataloader(self, input_data):
        # Dataloader
        self.bs = 1  # batch_size
        dataset = LoadImages(input_data, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)

        return dataset

    @smart_inference_mode()
    def standart_infer(self, input_data, save_to_file):
        camera_names = os.listdir(input_data)

        camera_names_lst = []
        img_name_lst = []
        is_dangers = []
        danger_percents = []

        images = glob(input_data + '*/*.jpg') + glob(input_data + '*/*.Png')
        print(len(images))

        t_start = time()
        # inference
        for camera in camera_names:
            dataset = self.get_dataloader(input_data=input_data + camera)

            self.model.warmup(imgsz=(1 if self.pt or self.model.triton else self.bs, 3, *self.imgsz))  # warmup
            seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

            for path, im, im0s, vid_cap, s in tqdm(dataset):
                camera_names_lst.append(camera)
                img_name_lst.append(path.split('/')[-1])

                with dt[0]:
                    im = torch.from_numpy(im).to(self.model.device)
                    im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim

                # Inference
                with dt[1]:
                    # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                    pred = self.model(im, augment=False, visualize=False)

                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

                # Process predictions
                is_danger = []
                danger_percent = []

                for i, det in enumerate(pred):  # per image
                    seen += 1
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)  # to Path

                    if len(det): 
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        det_objcts = reversed(det[:, :6]).cpu().numpy()

                        for det in det_objcts:
                            percent = self.get_alert(camera, det[:4])

                            if percent >= 0.15:
                                is_danger.append(1)
                            else:
                                is_danger.append(0)

                            danger_percent.append(percent)
                
                is_dangers.append(is_danger)
                danger_percents.append(danger_percent)
        
        # check speed
        t_work = time() - t_start
        LOGGER.info(f'FPS: {len(images) / t_work}')
        LOGGER.info(f'Total time: {t_work}')

        # write to csv
        final_dict = {'camera_name': camera_names_lst,
                      'image_name': img_name_lst,
                       'is_danger': is_dangers,
                       'danger_percents': danger_percents
                       }
        
        final_df = pd.DataFrame(final_dict)
        final_df.to_csv(save_to_file, index=False)

                            
    def get_alert(self, camera_name, xyxy):
        max_intersection = 0
        for poly in self.danger_zone_poly[camera_name]:
            intersection = self._box_poly_intersect(xyxy, poly)
            if intersection > max_intersection:
                max_intersection = intersection

        return max_intersection

    def _box_poly_intersect(self, box, poly):
        p1 = Point(box[0], box[1])
        p2 = Point(box[0], box[3])
        p3 = Point(box[2], box[3])
        p4 = Point(box[2], box[1])
        pointList = [p1, p2, p3, p4]
        box_coord = pointList
        boxp = Polygon(box_coord) # [[-1,-2], [-1,2], [1,2], [1,-2]]
        polyp = Polygon(poly)
        if boxp.intersects(polyp):
            return boxp.intersection(polyp).area / boxp.area # 0 - 1
        else:
            return 0


    @smart_inference_mode()
    def base_speed_test(self, input_data):
        dataset = self.get_dataloader(input_data=input_data)

        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else self.bs, 3, *self.imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

        t_start = time()
        for path, im, im0s, vid_cap, s in tqdm(dataset):
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                pred = self.model(im, augment=False, visualize=False)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path

                if len(det): 
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        t_work = time() - t_start
        LOGGER.info(f'FPS: {len(dataset) / t_work}')
        LOGGER.info(f'Total time: {t_work}')


if __name__ == '__main__':
    yp = YoloPredictor(weights='runs/train/small_1024/weights/best.pt')
    yp.standart_infer(input_data="cameras_data/final_test/",
                      save_to_file='submission2.csv')