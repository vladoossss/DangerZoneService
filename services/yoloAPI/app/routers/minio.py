from app.core.gateways.minio import MinioServer
from app.dependencies.minio import get_minio_instance

from fastapi import APIRouter, Depends, UploadFile, File
from typing import List
import cv2
from PIL import Image
import numpy as np
from io import BytesIO

router = APIRouter(
    prefix="/minio",
    tags=["minio"],
    dependencies=[Depends(get_minio_instance)]
    )

@router.post("")
def check_minio_connection(file_list: List[UploadFile] = File(...), server: MinioServer = Depends(get_minio_instance)):
    '''Check minio connection sends test.png to test bucket'''
    try:
        found = server.client.bucket_exists("test")
        if not found:
            server.client.make_bucket("test")
        else:
            print("Bucket already exists")

        img = [cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
                for file in file_list][0]
        
        img = Image.fromarray(img)
        out_img = BytesIO()
        img.save(out_img, format='png')
        out_img.seek(0) 

        server.client.put_object("test", "test.png", out_img, content_type='image/png', length=-1, part_size=10*1024*1024)
        # server.fput_object("test", "test.jpg", LOCAL_FILE_PATH)
        print("It is successfully uploaded to bucket")
    except Exception as e:
        raise e
    return 'Message sent successfully'