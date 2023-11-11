from glob import glob
from tqdm import tqdm
import shutil
import cv2
from sklearn.model_selection import train_test_split


png = glob('cameras_data/cameras/*/*.Png')
jpg = glob('cameras_data/cameras/*/*.jpg')
txt = glob('cameras_data/cameras/*/*.txt')

print('Всего фото:', len(jpg) + len(png))
print('Всего txt:', len(txt))

cameras_folder = glob('cameras_data/cameras/*')

for folder in tqdm(cameras_folder):
    images = glob(folder + '/*.jpg') + glob(folder + '/*.Png')

    train, test = train_test_split(images, test_size=0.2, random_state=42)
    val, test = train_test_split(test, test_size=0.5, random_state=42)

    for image in train:
        labels = image.replace('.jpg', '.txt')
        labels = labels.replace('.Png', '.txt')
        shutil.copy(image, "cameras_data/yolo_data/images/train")
        shutil.copy(labels, "cameras_data/yolo_data/labels/train")

    for image in val:
        labels = image.replace('.jpg', '.txt')
        labels = labels.replace('.Png', '.txt')
        shutil.copy(image, "cameras_data/yolo_data/images/val")
        shutil.copy(labels, "cameras_data/yolo_data/labels/val")

    for image in test:
        labels = image.replace('.jpg', '.txt')
        labels = labels.replace('.Png', '.txt')
        shutil.copy(image, "cameras_data/yolo_data/images/test")
        shutil.copy(labels, "cameras_data/yolo_data/labels/test")


print('images train:', len(glob('cameras_data/yolo_data/images/train/*')))
print('labels train:', len(glob('cameras_data/yolo_data/labels/train/*')))
print('images val:', len(glob('cameras_data/yolo_data/images/val/*')))
print('labels val:', len(glob('cameras_data/yolo_data/labels/val/*')))
print('images test:', len(glob('cameras_data/yolo_data/images/test/*')))
print('images test:', len(glob('cameras_data/yolo_data/labels/test/*')))


images = glob('/raid/nanosemantics/veklenko/yolov5/yolov5/cameras_data/yolo_data/images/train/*')
for image in images[:10]:
    im = cv2.imread(image)
    print(im.shape)