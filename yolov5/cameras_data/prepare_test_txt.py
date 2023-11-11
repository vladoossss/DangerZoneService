from glob import glob
from tqdm import tqdm
import shutil
import os
from sklearn.model_selection import train_test_split


cameras_folder = os.listdir('cameras_data/cameras')

for folder in tqdm(cameras_folder):
    txt = glob('cameras_data/cameras/' + folder + '/*.txt')

    train, test = train_test_split(txt, test_size=0.2, random_state=42)
    val, test = train_test_split(test, test_size=0.5, random_state=42)

    for image in test:
        if not os.path.exists('cameras_data/test_txt/' + folder):
            os.makedirs('cameras_data/test_txt/' + folder)
        shutil.copy(image, f"cameras_data/test_txt/{folder}")


print('images test:', len(glob('cameras_data/test_txt/*/*')))

