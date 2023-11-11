from PIL import Image
from io import BytesIO

from app.core.yolov5.detect_video import YoloBase
from app.dependencies.yolo import get_yolo_instance

from app.core.gateways.kafka import Kafka
from app.dependencies.kafka import get_kafka_instance

from app.core.gateways.minio import MinioServer
from app.dependencies.minio import get_minio_instance

from fastapi import APIRouter, Depends, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates

from typing import List, Optional
from app.utils.helpers import draw_poly, plot_one_box, base64EncodeImage, minio_post, kafka_post
import random
import cv2
import numpy as np

router = APIRouter(
    prefix="/images",
    tags=["images"],
    dependencies=[Depends(get_kafka_instance), Depends(get_minio_instance), Depends(get_yolo_instance)]
    )

colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(10)] #for bbox plotting

model_selection_options = ['name','DpR-Csp-uipv-ShV-V1', 'Pgp-com2-K-1-0-9-36', 'Pgp-lpc2-K-0-1-38', 
                            'Phl-com3-Shv2-9-K34', 'Php-Angc-K3-1', 'Php-Angc-K3-8', 
                            'Php-Ctm-K-1-12-56', 'Php-Ctm-Shv1-2-K3', 'Php-nta4-shv016309-k2-1-7', 
                            'Spp-210-K1-3-3-5', 'Spp-210-K1-3-3-6', 'Spp-K1-1-2-6']

templates = Jinja2Templates(directory = 'app/templates')

@router.get("")
def home(request: Request):
    ''' Returns html jinja2 template render for home page form
    '''

    return templates.TemplateResponse('home.html', {
            "request": request,
            "model_selection_options": model_selection_options,
        })

@router.post("")
def images_inference(request: Request,
                        file_list: List[UploadFile] = File(...), 
                        model_name: str = Form(...),
                        # img_size: int = Form(640),
                        minio: MinioServer = Depends(get_minio_instance),
                        kafka: Kafka = Depends(get_kafka_instance),
                        yolo: YoloBase = Depends(get_yolo_instance)):

    img_batch = [cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
                    for file in file_list]

    #create a copy that corrects for cv2.imdecode generating BGR images instead of RGB
    #using cvtColor instead of [...,::-1] to keep array contiguous in RAM
    img_batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_batch]

    json_results = []
    for im in img_batch_rgb:
        inp = np.expand_dims(im, 0)
        res = yolo.yolo(inp)
        json_results.extend(res)

    img_str_list = []
    #plot bboxes on the image
    for id, (img, bbox_list) in enumerate(zip(img_batch, json_results)):
        # print(bbox_list) # [{'class': 0, 'class_name': 'person', 'bbox': [1120, 356, 1289, 646], 'confidence': 0.29738280177116394}]
        send_requests = False
        request_template = {"image_name": "", "max_percent": 0, "text_message": "", "camera_name": ""}
        for obj in bbox_list:
            label = f'{obj["class_name"]} {obj["confidence"]:.2f}'
            if model_name in yolo.yolo.zones.keys():
                draw_poly(img, yolo.yolo.zones[model_name])
                inter_percent, text_message = yolo.yolo.get_alert(model_name, obj['bbox'])
                if inter_percent > 0.15:
                    plot_one_box(obj['bbox'], img, label=label, color=(0,0,255), line_thickness=3)
                    send_requests = True

                    if inter_percent > request_template["max_percent"]: # filling template 
                        request_template["max_percent"] = inter_percent
                        request_template["text_message"] = text_message
                    request_template["camera_name"] = model_name

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

        img_str_list.append(base64EncodeImage(img))

    #escape the apostrophes in the json string representation
    encoded_json_results = str(json_results).replace("'",r"\'").replace('"',r'\"')

    return templates.TemplateResponse('show_results.html', {
            'request': request,
            'bbox_image_data_zipped': zip(img_str_list,json_results), #unzipped in jinja2 template
            'bbox_data_str': encoded_json_results,
        })