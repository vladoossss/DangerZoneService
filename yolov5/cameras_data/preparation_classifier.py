from glob import glob
from tqdm import tqdm
import shutil
import cv2
import os
from sklearn.model_selection import train_test_split


cameras_folder = os.listdir('cameras_data/cameras')

for folder in tqdm(cameras_folder):
    images = glob('cameras_data/cameras/' + folder + '/*.jpg') + glob('cameras_data/cameras/' + folder + '/*.Png')

    train, test = train_test_split(images, test_size=0.2, random_state=42)
    val, test = train_test_split(test, test_size=0.5, random_state=42)

    for image in train:
        if not os.path.exists('cameras_data/resnet_data/train/' + folder):
            os.makedirs('cameras_data/resnet_data/train/' + folder)
        shutil.copy(image, f"cameras_data/resnet_data/train/{folder}")

    for image in val:
        if not os.path.exists('cameras_data/resnet_data/val/' + folder):
            os.makedirs('cameras_data/resnet_data/val/' + folder)
        shutil.copy(image, f"cameras_data/resnet_data/val/{folder}")

    for image in test:
        if not os.path.exists('cameras_data/resnet_data/test/' + folder):
            os.makedirs('cameras_data/resnet_data/test/' + folder)
        shutil.copy(image, f"cameras_data/resnet_data/test/{folder}")


# print('images train:', len(glob('/raid/nanosemantics/veklenko/yolov5/yolov5/cameras_data/resnet_data/train/*/*')))
# print('images val:', len(glob('/raid/nanosemantics/veklenko/yolov5/yolov5/cameras_data/resnet_data/val/*/*')))
# print('images test:', len(glob('/raid/nanosemantics/veklenko/yolov5/yolov5/cameras_data/resnet_data/test/*/*')))

