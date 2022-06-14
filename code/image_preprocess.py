from PIL import Image

#necessary to have openslide working
import os



import math
import pandas as pd



def downscale_from_manifest(manifestdirectory,svsdirectory,outputDirectory,scale,store_together,openslide_path):
    svsdirectory = svsdirectory.replace("\\","/")
    outputDirectory = outputDirectory.replace("\\", "/")
    ###############################################
    #to fix the issues with openslide in windows
    if openslide_path:
        os.add_dll_directory(openslide_path)
        os.environ['PATH'] = openslide_path + ";" + os.environ['PATH']
    import openslide
    ###############################################
    # make bigger the maximum size of file admited
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
    def remove_prefix(text, prefix):
        if text.startswith(prefix):
            return text[len(prefix):]
        return text

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
        return img  # , large_w, large_h, new_w, new_h


    ###############################################
    if (not manifestdirectory) or os.path.isfile(manifestdirectory) == True:
        not_processed = []
        if manifestdirectory and os.path.isfile(manifestdirectory) == True:
            manifest = pd.read_csv(manifestdirectory,sep = "\t")
            dir = list(manifest["id"])
            filename = list(manifest["filename"])
        else:
            dir = []
            filename = []
            for root, dirs, files in os.walk(svsdirectory):
                for f in files:
                    filename.append(f)
                    root = root.replace("\\", "/")
                    dir.append(remove_prefix(remove_prefix(root,"/"),svsdirectory))
            # print(dir)
            # print(filename)
            # print(len(dir))
            # print(len(filename))
        for i in range(0,len(dir)):
            #path = os.path.join(svsdirectory,dir[i],filename[i])
            path = (svsdirectory + "/" + dir[i] + "/" + filename[i])
            #print(filename[i])
            if store_together:
                out_path = os.path.join(outputDirectory,remove_suffix(filename[i],".svs")) + ".png"
            else:
                out_path = os.path.join((outputDirectory +dir[i]),remove_suffix(filename[i],".svs")) + ".png"
            #print(path)
            #print(os.path.join((outputDirectory +dir[i]),remove_suffix(filename[i],".svs")) + ".png")
            if (os.path.isfile(path) == True) and (os.path.isfile(out_path) == False) and path.endswith(".svs"):
                try:
                    img = downscale(path,scale)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    img.save(out_path)
                    print("image saved as: ", out_path)
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
