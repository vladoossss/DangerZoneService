import cv2
import base64
import numpy as np
from datetime import datetime


def results_to_json(results, model):
    ''' Converts yolo model output to json (list of list of dicts)'''
    return [
                [
                    {
                    "class": int(pred[5]),
                    "class_name": model.model.names[int(pred[5])],
                    "bbox": [int(x) for x in pred[:4].tolist()], #convert bbox results to int from float
                    "confidence": float(pred[4]),
                    }
                for pred in result
                ]
            for result in results.xyxy
            ]

def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Directly copied from: https://github.com/ultralytics/yolov5/blob/cd540d8625bba8a05329ede3522046ee53eb349d/utils/plots.py
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_poly(cvimage, poly):
    for segment in poly: # in case there are more than one segmrnt
        vertices = segment.reshape((-1, 1, 2))
        cv2.polylines(cvimage, [vertices], isClosed=True, color=(255, 0, 0), thickness=2)

def base64EncodeImage(img):
    ''' Takes an input image and returns a base64 encoded string representation of that image (jpg format)'''
    _, im_arr = cv2.imencode('.jpg', img)
    im_b64 = base64.b64encode(im_arr.tobytes()).decode('utf-8')

    return im_b64

def minio_post(image_bytes, minio, bucket_name='cameras_events'):
            try:
                found = minio.client.bucket_exists(bucket_name)
                if not found:
                    minio.client.make_bucket(bucket_name)
                else:
                    print("Bucket already exists")

                current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
                im_name = f'{current_datetime}.jpg'

                minio.client.put_object(bucket_name, im_name, image_bytes, content_type='image/jpeg', length=-1, part_size=10*1024*1024)
                # server.fput_object("test", "test.jpg", LOCAL_FILE_PATH)
                print("It is successfully uploaded to bucket")
            except Exception as e:
                raise e
            return im_name

def kafka_post(dict_, server):
    try:
        topic_name = server._topic
        server.producer.send(topic_name, dict_)
    except Exception as e:
        raise e
    return 'Message sent successfully'