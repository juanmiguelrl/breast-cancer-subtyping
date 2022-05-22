import requests
import json
import pandas as pd


def download_manifest_from_GDC(output_file,projects,name_restrictions,files_data_format="svs",
                               experimental_strategy="Tissue Slide",endpoint='https://api.gdc.cancer.gov/files'):
    fields = [
        "file_name",
        "md5sum",
        "file_size",
        "state"
    ]

    fields = ",".join(fields)

    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": projects
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "files.data_format",
                    "value": files_data_format
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "files.experimental_strategy",
                    "value": experimental_strategy
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "files.file_name",
                    "value": name_restrictions
                }
            }
        ]
    }

    # With a GET request, the filters parameter needs to be converted
    # from a dictionary to JSON-formatted string

    params = {
        "filters": json.dumps(filters),
        "fields": fields,
        "format": "TSV",
        "size": "50000"
    }

    response = requests.get(endpoint, params=params)

    file = output_file
    # open text file
    text_file = open(file, "w+")

    # write string to file
    text_file.write(response.content.decode("utf-8"))

    # close file
    text_file.close()

    # print(response.content)

    data = pd.read_csv(file, sep="\t")
    # with tthose changes the dataframe will have exactly the same format as downloading the manifest from the GDC webpage
    data.rename(columns={"file_name": "filename", "md5sum": "md5", "file_size": "size"}, inplace=True)
    data.set_index("id", inplace=True)
    column_names = ["filename", "md5", "size", "state"]
    data = data.reindex(columns=column_names)
    data.to_csv(output_file, sep="\t")
    return


def download_data(output_file, manifest_path, projects, name_restrictions, fields_dictionary, experimental_strategy,
                  expand,files_data_format="svs", endpoint='https://api.gdc.cancer.gov/cases'):
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": projects
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "files.data_format",
                    "value": files_data_format
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "files.experimental_strategy",
                    "value": experimental_strategy
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "files.file_name",
                    "value": name_restrictions
                }
            }
        ]
    }

    params = {
        "filters": json.dumps(filters),
        # "fields": fields,
        "expand": expand,
        "format": "TSV",
        "size": "50000"
    }

    response = requests.get(endpoint, params=params)

    file = output_file
    # open text file
    text_file = open(file, "w+")
    # write string to file
    text_file.write(response.content.decode("utf-8"))
    # close file
    text_file.close()

    clinical = pd.read_csv(file, sep="\t")
    # create empty dataframe
    definitive = pd.DataFrame()

    def dic(index, fields_dictionary):
        d = {"case_id": clinical.loc[index, "case_id"],
             "slide_ids": clinical.loc[index, column].lower(),
             "submitter_id": clinical.loc[index, "submitter_id"]
             }
        for key, value in fields_dictionary.items():
            d.update({key: clinical.loc[index, value]})
        return d

    row_list = []
    for index, row in clinical.iterrows():
        for column in clinical.columns:
            if column.startswith("slide_ids"):
                if not pd.isnull(clinical.loc[index, column]):
                    row_list.append(dic(index, fields_dictionary))
    definitive = pd.DataFrame(row_list)

    manifest = pd.read_csv(manifest_path, sep="\t")
    manifest["slide_ids"] = manifest["filename"].str.split(".").str[1].str.lower()

    definitive = pd.merge(definitive, manifest, on="slide_ids", how="inner")
    definitive.to_csv(output_file, sep="\t",index=False)
    return definitive

def download_data_with_R(executable,r_path,r_script_path,arguments):
    """ Download data using R
    """
    import subprocess
    import os
    if executable:
        exec_name = r_path
    else:
        exec_name = "Rscript"
    print([exec_name, r_script_path.replace("\\", "/")])
    pipe = subprocess.run([exec_name, r_script_path.replace("\\", "/")]+arguments,shell=True)
    return

def download_images_from_manifest(manifest_file,output_dir,executable=False,executable_path_file=None,command_for_gdc_client="gdc-client"):
    """ Download images from manifest file
        calling the gdc transfer tool
    """
    import subprocess
    import os
    if not executable:
        executable_path_file = "gdc-client"
        exec_name = command_for_gdc_client
    else:
        path = executable_path_file.rsplit('\\',1)[0]
        path.replace("\\", "/")
        os.chdir(path)
        print(os.getcwd())
        exec_name = executable_path_file.rsplit('\\')[-1]
    #print("\""+exec_name+"\"" + " download " + " -m", manifest_file, " -d", output_dir)
    print([exec_name, "download", "-m", manifest_file.replace("\\", "/"), "-d", output_dir.replace("\\", "/")])
    pipe = subprocess.run([exec_name, "download" , "-m", manifest_file.replace("\\", "/"), "-d", output_dir.replace("\\", "/")],shell=True)
    #text = pipe.communicate()[0]
    #print(text)

def join_data(data):
    """ Join dataframes
    """
    for element in data:
        df1 = pd.read_csv(element["df1_path"],sep=element["sep1"])
        df2 = pd.read_csv(element["df2_path"],sep=element["sep2"])
        df = pd.merge(df1,df2,left_on=element["left_on"],right_on=element["right_on"],how=element["join_type"])
        df.to_csv(element["output_file"],sep="\t",index=False)
    return df

#pd.merge(data, pam50[["BRCA_Subtype_PAM50"]], left_on="case_submitter_id",right_on="patient", right_index=True,how="left")


import tensorflow as tf
import numpy as np

# class DataGenerator(tf.keras.utils.Sequence):
#     def __init__(self, df, x_col, y_col=None, batch_size=32, num_classes=None, shuffle=True):
#         self.batch_size = batch_size
#         self.df = dataframe
#         self.indices = self.df.index.tolist()
#         self.num_classes = num_classes
#         self.shuffle = shuffle
#         self.x_col = x_col
#         self.y_col = y_col
#         self.on_epoch_end()
#
#     def __len__(self):
#         return len(self.indices) // self.batch_size)
#
#         def __getitem__(self, index):
#             index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
#             batch = [self.indices[k] for k in index]
#
#             X, y = self.__get_data(batch)
#             return X, y
#
#         def on_epoch_end(self):
#             self.index = np.arange(len(self.indices))
#             if self.shuffle == True:
#                 np.random.shuffle(self.index)
#
#         def __get_data(self, batch):
#             X =  # logic
#             y =  # logic
#
#             for i, id in enumerate(batch):
#                 X[i,] =  # logic
#                 y[i] =  # labels
#
#             return X, y
#
#
# from keras.preprocessing.image import Iterator, ImageDataGenerator
#
#
# import os
# import random
# import pandas as pd
#
# def generator(image_dir, csv_dir, batch_size):
#     i = 0
#     image_file_list = os.listdir(image_dir)
#     while True:
#         batch_x = {'images': list(), 'other_feats': list()}  # use a dict for multiple inputs
#         batch_y = list()
#         for b in range(batch_size):
#             if i == len(image_file_list):
#                 i = 0
#                 random.shuffle(image_file_list)
#             sample = image_file_list[i]
#             image_file_path = sample[0]
#             csv_file_path = os.path.join(csv_dir,
#                                          os.path.basename(image_file_path).replace('.png', '.csv'))
#             i += 1
#             image = preprocess_image(cv2.imread(image_file_path))
#             csv_file = pd.read_csv(csv_file_path)
#             other_feat = preprocess_feats(csv_file)
#             batch_x['images'].append(image)
#             batch_x['other_feats'].append(other_feat)
#             batch_y.append(csv_file.loc[image_name, :]['class'])
#
#         batch_x['images'] = np.array(batch_x['images'])  # convert each list to array
#         batch_x['other_feats'] = np.array(batch_x['other_feats'])
#         batch_y = np.eye(num_classes)[batch['labels']]
#         yield batch_x, batch_y
#
#
#
# class JoinedGen(tf.keras.utils.Sequence):
#     def __init__(self, img_gen, dataframe, target_gen):
#         self.img_gen = img_gen
#         self.dataframe = dataframe
#
#         assert len(input_gen1) == len(input_gen2) == len(target_gen)
#
#     def __len__(self):
#         return len(self.gen1)
#
#     def __getitem__(self, i):
#         x1 = self.gen1[i]
#         x2 = self.gen2[i]
#         y = self.gen3[i]
#
#         return [x1, x2], y
#
#     def on_epoch_end(self):
#         self.gen1.on_epoch_end()
#         self.gen2.on_epoch_end()
#         self.gen3.on_epoch_end()
#         self.gen2.index_array = self.gen1.index_array
#         self.gen3.index_array = self.gen1.index_array
#
#
