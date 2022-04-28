import glob
import shutil
import os

def get_filename(path):
    return path.split('\\')[-1]

def store_images_together(destination_path,input_dir):
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    for img in glob.glob(input_dir):
        shutil.copy(img, destination_path+"\\"+get_filename(img))