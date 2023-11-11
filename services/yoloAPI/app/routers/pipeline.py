from app.core.gateways.kafka import Kafka
from app.dependencies.kafka import get_kafka_instance

from app.core.gateways.minio import MinioServer
from app.dependencies.minio import get_minio_instance

from app.core.yolov5.detect_video import YoloBase
from app.dependencies.yolo import get_yolo_instance

from fastapi import APIRouter, Depends, UploadFile, File, Request
from tempfile import NamedTemporaryFile

from app.utils.helpers import plot_one_box, draw_poly, kafka_post, minio_post

from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import random

router = APIRouter(
    prefix="/pipeline",
    tags=["pipeline"],
    dependencies=[Depends(get_kafka_instance), Depends(get_minio_instance), Depends(get_yolo_instance)]
    )

colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(10)] #for bbox plotting

@router.post("/")
def send(request: Request,
        # camera: Camera,
        file: UploadFile = File(...),
        minio: MinioServer = Depends(get_minio_instance),
        kafka: Kafka = Depends(get_kafka_instance),
        yolo: YoloBase = Depends(get_yolo_instance)
        ):
    
    def process_video(video, model):
        cap = cv2.VideoCapture(video)

        cnt = 0
        frame_timeout=150
        frame_timeout_bool = True

        while cap.isOpened():
            ret, frame = cap.read()
            frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if frame_timeout_bool:
                img_batch = np.expand_dims(frame, axis=0)

                json_results = model.yolo(img_batch) #TODO check need RGB or BGR

                camera_name = 'demo' #camera.dict()['camera_name']

                for id, (img, bbox_list) in enumerate(zip(img_batch, json_results)):
                    # print(bbox_list) # [{'class': 0, 'class_name': 'person', 'bbox': [1120, 356, 1289, 646], 'confidence': 0.29738280177116394}]
                    send_requests = False
                    request_template = {"image_name": "", "max_percent": 0, "text_message": "", "camera_name": ""}
                    for obj in bbox_list:
                        label = f'{obj["class_name"]} {obj["confidence"]:.2f}'
                        if camera_name in model.yolo.zones.keys():
                            draw_poly(img, model.yolo.zones[camera_name])
                            inter_percent, text_message = model.yolo.get_alert(camera_name, obj['bbox'])
                            if inter_percent > 0.15:
                                plot_one_box(obj['bbox'], img, label=label, color=(0,0,255), line_thickness=3)
                                send_requests = True

                                if inter_percent > request_template["max_percent"]: # filling template 
                                    request_template["max_percent"] = inter_percent
                                    request_template["text_message"] = text_message
                                request_template["camera_name"] = camera_name

                            else:
                                plot_one_box(obj['bbox'], img, label=label, color=(0,255,0), line_thickness=3)
                        else:
                            plot_one_box(obj['bbox'], img, label=label, 
                                color=colors[int(obj['class'])], line_thickness=3)

                    if send_requests:
                        imgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        out_img = BytesIO()
                        imgb.save(out_img, format='jpeg')
                        out_img.seek(0)

                        im_name = minio_post(out_img, minio)
                        request_template['image_name'] = im_name

                        message = kafka_post(request_template, kafka)
                        print(message)

                        cnt = frame_num
                        frame_timeout_bool =False

            if frame_num == cnt + frame_timeout:
                frame_timeout_bool = True

        cap.release()

    temp = NamedTemporaryFile(delete=False)
    try:
        contents = file.file.read()
        with temp as f:
            f.write(contents);
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    res = process_video(temp.name, yolo)  # Pass temp.name to VideoCapture()
        
    return res
