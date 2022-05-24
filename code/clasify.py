# %%
import os
import numpy as np #para numeros
import pandas as pd #para matrices
from sklearn.impute import SimpleImputer #para missing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #para codificar categoricos
#from keras.utils import np_utils


from os import listdir
from os.path import isfile, join

from sklearn.model_selection import train_test_split

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
    data.to_csv(output_file, sep="\t",index=False)
    return data

def clasify_multiple(list_of_dictionaries):
    for PARAMS in list_of_dictionaries:
        clasify_img = {"simplify_stage": False}
        clasify_img.update(PARAMS)
        clasify_images(clasify_img["input"], clasify_img["imgdir"], clasify_img["classification"], clasify_img["output_file"],clasify_img["simplify_stage"])

#####################################################################
#if directory does not exist, create it
def makedirectory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Move .png files into train and test directories
def copyFiles(sourceDir, destDir, fileList):
    for file in fileList:
        shutil.copy(os.path.join(sourceDir, file+".png"), destDir)

def clasify_images_old(input,imgdir,sourceDir,newDirTOsplitImages):
    data = pd.read_csv(input, sep='\t', header=0)

    data["stage"] = data["ajcc_pathologic_stage"].apply(simplify)
    #data.to_csv(input, sep='\t', index=False)
    #drop rows with stage 0 (which is the stage previously added in smplify() for the missing values)
    data = data[data["stage"] != 0]
    nsalidas = len(set(data["stage"]))
    data["target_enc"] = LabelEncoder().fit_transform(data.stage)
    #Y_hot = np_utils.to_categorical(data["target_enc"])
    X = data
    #Y = Y_hot

    #to only take the files which are in the folder and not in the subfolders (so the discarded images are not taken)
    onlyfiles = [rchop(f,".png") for f in listdir(imgdir) if isfile(join(imgdir, f))]

    #drop rows of X which are not in onlyfiles
    Xold = X.copy()
    X = X[X["filename"].apply(cut_svs).isin(onlyfiles)]
    X["filename"] = X["filename"].apply(cut_svs)

    trainX, testX = train_test_split(X, test_size=0.25,stratify=X["stage"])
    #print(trainX.shape)
    #print(trainX.head)
    #print(testX.shape)
    #print(testX.head)

    #divide in the 4 classes of stage in x in the train and test sets
    trainC1 = list(trainX.loc[trainX['stage'] == 1]["filename"])
    #trainC1 = [i[0:12] for i in trainC1]
    trainC2 = list(trainX.loc[trainX['stage'] == 2]["filename"])
    #trainC2 = [i[0:12] for i in trainC2]
    trainC3 = list(trainX.loc[trainX['stage'] == 3]["filename"])
    #trainC3 = [i[0:12] for i in trainC3]
    trainC4 = list(trainX.loc[trainX['stage'] == 4]["filename"])
    #trainC4 = [i[0:12] for i in trainC4]

    testC1 = list(testX.loc[testX['stage'] == 1]["filename"])
    #testC1 = [i[0:12] for i in testC1]
    testC2 = list(testX.loc[testX['stage'] == 2]["filename"])
    #testC2 = [i[0:12] for i in testC2]
    testC3 = list(testX.loc[testX['stage'] == 3]["filename"])
    #testC3 = [i[0:12] for i in testC3]
    testC4 = list(testX.loc[testX['stage'] == 4]["filename"])
    #testC4 = [i[0:12] for i in testC4]
    #print(trainC1); print(trainC2); print(testC1); print(testC2)
    #print(len(trainC1)); print(len(trainC2)); print(len(testC1)); print(len(testC2))

    # Create directories to distribute Images .png
    #os.mkdir(newDirTOsplitImages)
    makedirectory(newDirTOsplitImages)
    trainDir = os.path.join(newDirTOsplitImages, 'train');
    makedirectory(trainDir)
    trainDirC1 = os.path.join(trainDir, 'class1');
    makedirectory(trainDirC1)
    trainDirC2 = os.path.join(trainDir, 'class2');
    makedirectory(trainDirC2)
    trainDirC3 = os.path.join(trainDir, 'class3');
    makedirectory(trainDirC3)
    trainDirC4 = os.path.join(trainDir, 'class4');
    makedirectory(trainDirC4)

    testDir = os.path.join(newDirTOsplitImages, 'test') ;
    makedirectory(testDir)
    testDirC1 = os.path.join(testDir, 'class1');
    makedirectory(testDirC1)
    testDirC2 = os.path.join(testDir, 'class2');
    makedirectory(testDirC2)
    testDirC3 = os.path.join(testDir, 'class3');
    makedirectory(testDirC3)
    testDirC4 = os.path.join(testDir, 'class4');
    makedirectory(testDirC4)
    
    # copy .png files into train and test directories
    copyFiles(sourceDir, trainDirC1, trainC1)
    copyFiles(sourceDir, trainDirC2, trainC2)
    copyFiles(sourceDir, trainDirC3, trainC3)
    copyFiles(sourceDir, trainDirC4, trainC4)

    copyFiles(sourceDir, testDirC1, testC1)
    copyFiles(sourceDir, testDirC2, testC2)
    copyFiles(sourceDir, testDirC3, testC3)
    copyFiles(sourceDir, testDirC4, testC4)


#####################################################################
#simplify the the pathologic state in 4 stages
def indicate_NaN(x):
    #check if x is string
    if not isinstance(x, str):
        return "NaN"



def clasify_images_oldv2(input,imgdir,sourceDir,newDirTOsplitImages,classification):
    data = pd.read_csv(input, sep='\t', header=0)
    data["stage"] = data[classification].apply(indicate_NaN)
    # drop rows with stage NaN (which is the stage previously added in indicate_NaN() for the missing values)
    data = data[data["stage"] != "NaN"]
    data["target_enc"] = LabelEncoder().fit_transform(data.stage)
    X = data

    #to only take the files which are in the folder and not in the subfolders (so the discarded images are not taken)
    onlyfiles = [rchop(f,".png") for f in listdir(imgdir) if isfile(join(imgdir, f))]

    #drop rows of X which are not in onlyfiles
    Xold = X.copy()
    X = X[X["filename"].apply(cut_svs).isin(onlyfiles)]
    X["filename"] = X["filename"].apply(cut_svs)

    trainX, testX = train_test_split(X, test_size=0.25, stratify=X[classification])

    makedirectory(newDirTOsplitImages)
    trainDir = os.path.join(newDirTOsplitImages, 'train')
    makedirectory(trainDir)
    testDir = os.path.join(newDirTOsplitImages, 'test')
    makedirectory(testDir)

    for element in X[classification].unique():
        train = list(trainX.loc[trainX[classification] == element]["filename"])
        test = list(testX.loc[testX[classification] == element]["filename"])
        #make a directory for each subtype
        trainsubtype = os.path.join(trainDir, element)
        makedirectory(trainsubtype)
        testsubtype = os.path.join(testDir, element)
        makedirectory(testsubtype)
        #copy the files into the subtype directory
        copyFiles(sourceDir, trainsubtype, train)
        copyFiles(sourceDir, testsubtype, test)





