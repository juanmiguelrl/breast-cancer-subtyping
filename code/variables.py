import os

manifestdirectory = "/mnt/netapp2/Store_uni/home/ulc/co/jrl/imagenes/manifest.txt"
svsdirectory = "/mnt/netapp2/Store_uni/home/ulc/co/jrl/imagenes/https:/api.gdc.cancer.gov/data/"
outputDirectory = "/mnt/netapp2/Store_uni/home/ulc/co/jrl/imagenes/mini_output_images"
scale = 64

class downscale:
    manifest_path = "/mnt/netapp2/Store_uni/home/ulc/co/jrl/imagenes/manifest.txt"
    svsdirectory = "/mnt/netapp2/Store_uni/home/ulc/co/jrl/imagenes/https:/api.gdc.cancer.gov/data/"
    outputDirectory = "/mnt/netapp2/Store_uni/home/ulc/co/jrl/imagenes/mini_output_images"
    scale = 32

class store:
    # to store all the reduced images in one folder (to revise them visually without going through a lot of folders)
    destination_path = "D:\\onedriveudc\\OneDrive - Universidade da Coruña\\tfg\\python\\data\\images32_together"
    input_dir = "D:\\onedriveudc\\OneDrive - Universidade da Coruña\\tfg\\python\\data\\images32\\*\\*"

class filter:
    input_dir = "D:\\onedriveudc\\OneDrive - Universidade da Coruña\\tfg\\python\\data\\images32_together\\*"
    destination_path = "D:\\onedriveudc\\OneDrive - Universidade da Coruña\\tfg\\python\\data\\filtered_images32\\"
    prefix = "D:\\onedriveudc\\OneDrive - Universidade da Coruña\\tfg\\python\\data\\images32_together\\"


class Clasify_var:
    input = "D:\\onedriveudc\\OneDrive - Universidade da Coruña\\tfg\\python\\data\\slide_stage_relation.csv"
    imgdir = "D:\\onedriveudc\\OneDrive - Universidade da Coruña\\tfg\\python\\data\\crop_resize_images32"
    sourceDir = imgdir + "\\"
    newDirTOsplitImages  = "D:\\onedriveudc\\OneDrive - Universidade da Coruña\\tfg\\python\\data\\slides_crop_split32"

class ann_var:
    #directories
    inputDir = 'D:\\onedriveudc\\OneDrive - Universidade da Coruña\\tfg\\python\\data\\slides_crop_split32\\'
    trainDir = os.path.join(inputDir, 'train\\')
    valDir = os.path.join(inputDir, 'test\\')
    logdir = "D:\\onedriveudc\\OneDrive - Universidade da Coruña\\tfg\\python\\logs\\fit\\"
    batch_size = 32
    epochs = 10
    n_gpus = 1