from PIL import Image

#necessary to have openslide working
import os
os.add_dll_directory('D://onedriveudc//OneDrive - Universidade da Coruña//tfg//python//openslide//bin')
openslide_path = 'D://onedriveudc//OneDrive - Universidade da Coruña//tfg//python//openslide//bin'
os.environ['PATH'] = openslide_path + ";" + os.environ['PATH']

import openslide
import math
import pandas as pd

#make bigger the maximum size of file admited
Image.MAX_IMAGE_PIXELS = 10000000000


def getslide(path):
    slide = openslide.open_slide(path)
    return slide

###############
def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string
###############



def downscale(path, SCALE_FACTOR):
    slide = openslide.open_slide(path)
    large_w, large_h = slide.dimensions
    new_w = math.floor(large_w / SCALE_FACTOR)
    new_h = math.floor(large_h / SCALE_FACTOR)
    level = slide.get_best_level_for_downsample(SCALE_FACTOR)
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), Image.BILINEAR)
    return img#, large_w, large_h, new_w, new_h

def downscale_from_manifest(manifestdirectory,svsdirectory,outputDirectory,scale):
    if os.path.isfile(manifestdirectory) == True:
        manifest = pd.read_csv(manifestdirectory,sep = "\t")
        dir = list(manifest["id"])
        filename = list(manifest["filename"])
        not_processed = []
        for i in range(0,len(dir)):
            #path = os.path.join(svsdirectory,dir[i],filename[i])
            path = (svsdirectory + "/" + dir[i] + "/" + filename[i])
            #print(filename[i])
            out_path = (outputDirectory + "/" + dir[i] + "/" + remove_suffix(filename[i],".svs") + ".png")
            #print(path)
            if (os.path.isfile(path) == True) and (os.path.isfile(out_path) == False):
                try:
                    img = downscale(path,scale)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    img.save(out_path)
                    print("image saved")
                    print("found file: " + path + "\n")
                #else:
                    #print("not found file: " + path + "\n")
                except Exception as e:
                    print("Error: ", Exception)
                    print("In folder: " + dir[i] + " with file " + filename[i])
                    not_processed.append(path)
            else:
                not_processed.append(path)
        #print("Files that could not be processed:")
        #for element in not_processed:
            #print(element)
    else:
        print("manifest directory not found")

    return
