import glob
import shutil
import os

def get_filename(path):
    return path.split('\\')[-1]

def store_images_together(destination_path,input_dir):
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    for root, dirs, files in os.walk(input_dir):
        for img in files:
            path_file = os.path.join(root, img)
            shutil.copy(path_file , destination_path)