# %%
import os
import pandas as pd #for the matrices
from os import listdir
from os.path import isfile, join
import shutil



#simplify the the pathologic stage in 4 stages
def simplify_stage(x):
    #check if x is string
    if isinstance(x, str):
        if "Stage IV" in x:
            return "Stage IV"
        elif "Stage III" in x:
            return "Stage III"
        elif "Stage II" in x:
            return "Stage II"
        elif "Stage I" in x:
            return "Stage I"
        else:
            return "unknown"
    else:
        return 0

def simplify(data,classification,target,simplify):
    if classification == "stage":
        data[target] = data["stage"].apply(simplify_stage)
        data = data[data[target] != "unknown"]
    else:
        data[target] = data[classification]
    if simplify:
        data["stage_simplified"] = data["stage"].apply(simplify_stage)
    return data


def rchop(s, suffix):
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    return s

def cut_svs(x):
    return rchop(x,".svs")

#simplify the the pathologic state in 4 stages
def indicate_NaN(x):
    #check if x is string
    if pd.isnull(x):
        return "NaN"

def clasify_images(input,imgdir,classification,output_file,simplify_stage):
    data = pd.read_csv(input, sep='\t', header=0)
    #data[classification] = data[classification].apply(indicate_NaN)
    # drop rows with stage NaN (which is the stage previously added in indicate_NaN() for the missing values)
    data = data[data[classification].isnull() != True]
    #simplify classes
    target = "target"
    data = simplify(data,classification,target,simplify_stage)
    #to only take the files which are in the folder and not in the subfolders (so the discarded images are not taken)
    onlyfiles = [rchop(f,".png") for f in listdir(imgdir) if isfile(join(imgdir, f))]
    data["filename"] = data["filename"].apply(cut_svs)
    data = data[data["filename"].isin(onlyfiles)]
    data["filepath"] = imgdir + os.path.sep + data["filename"].astype(str) + ".png"
    data["filepath"] = data["filepath"].replace("/", "\\")
    outdir = os.path.dirname(output_file)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    data.to_csv(output_file, sep="\t",index=False)
    return data

def clasify_multiple(list_of_dictionaries):
    for PARAMS in list_of_dictionaries:
        clasify_img = {"simplify_stage": False}
        clasify_img.update(PARAMS)
        clasify_images(clasify_img["input"], clasify_img["imgdir"], clasify_img["classification"], clasify_img["output_file"],clasify_img["simplify_stage"])

def dataframe_from_directory(imgdir,extension,output_file):
    data = pd.DataFrame([])
    #for each folder of the directory, it classifies the images into a dataframe
    for folder in os.listdir(imgdir):
        if os.path.isdir(os.path.join(imgdir, folder)):
            #print(folder)
            #for each file in the folder, it adds the filepath and the classification
            for filename in os.listdir(os.path.join(imgdir, folder)):
                path = os.path.join(imgdir, folder)
                if filename.endswith(extension):
                    data = data.append(pd.DataFrame({"filepath": [os.path.join(path,filename)], "target": [folder]}),ignore_index = True)
    outdir = os.path.dirname(output_file)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    data.to_csv(output_file, sep="\t",index=False)

def dataframe_from_directory_multiple(list_of_dictionaries):
    for PARAMS in list_of_dictionaries:
        clasify_img = {"simplify_stage": False}
        clasify_img.update(PARAMS)
        dataframe_from_directory(clasify_img["imgdir"], clasify_img["extension"], clasify_img["output_file"])


#####################################################################
#if directory does not exist, create it
def makedirectory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Move .png files into train and test directories
def copyFiles(sourceDir, destDir, fileList):
    for file in fileList:
        shutil.copy(os.path.join(sourceDir, file+".png"), destDir)


#####################################################################
#simplify the the pathologic state in 4 stages
def indicate_NaN(x):
    #check if x is string
    if not isinstance(x, str):
        return "NaN"
