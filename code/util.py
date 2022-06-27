import requests
import json
import pandas as pd
import sklearn
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import sklearn.model_selection
import numpy as np
import subprocess

def calculate_class_weights(train_generator):
    counter = Counter(train_generator.classes)
    max_val = float(max(counter.values()))
    class_weight = {class_id: max_val / num_images for class_id, num_images in counter.items()}
    return class_weight

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
    if executable:
        exec_name = r_path
    else:
        exec_name = "Rscript"
    print([exec_name, r_script_path.replace("\\", "/")]+arguments)
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


def modify_target(dictionary):
    def group_classes(df, dict):
        data = df.copy()
        # drop rows with the values specified
        if "drop" in dict:
            for element in dict["drop"]:
                data = data[data["target"].str.contains(element) == False]
        data["original_target"] = data["target"]

        def group(x):
            for key, element in dict["new_group_list"].items():
                if x in element:
                    return key
            return dict["other_name"]

        # make the new groups for the target
        if dict["new_group"]:
            data["target"] = data["target"].apply(group)
        if dict["remove_other"]:
            data = data[data["target"] != dict["other_name"]]
        return data

    dataframe = pd.read_csv(dictionary["input"], sep="\t")

    data = group_classes(dataframe, dictionary)

    data.to_csv(dictionary["output"], sep="\t", index=False)


def modify_multiple_targets(list_of_dictionaries):
    for element in list_of_dictionaries:
        modify_target(element)

################################################################################

#process the data for the model which uses the clinical data
def process_clinical_data(dataframe,parameters):
    cs = MinMaxScaler()
    data = dataframe.copy()
    #data.set_index("filepath", inplace=True)
    # data = data[parameters["continuos"] + parameters["categorical"] + ["target"]]
    for column in data:
        if column in parameters["continuos"]:
            #data[column].replace(np.nan, 0.1)
            #data[column] = data[column].apply(removenan)
            data[column].fillna(0, inplace=True)
        if column in parameters["categorical"]:
            data[column].replace(np.nan, "unknown")
    data = data[parameters["continuos"] + parameters["categorical"] + ["target","filepath"]]
    data[parameters["continuos"]] = cs.fit_transform(data[parameters["continuos"]])
    data = pd.get_dummies(data, columns=parameters["categorical"])

    # target = dataframe.copy()
    # #target.set_index("filepath", inplace=True)
    # target = target["target"]
    # target = pd.get_dummies(target, columns=["target"])

    return data#,target

def multiple_process_clinical_data(list_of_dictionaries):
    for parameters in list_of_dictionaries:
        process_clinical_data(pd.read_csv(parameters["dataframe"], sep='\t', header=0),parameters["clinical_columns"]).to_csv(parameters["output_dataframe"], sep='\t', index=False)
    return

def split_Dataframe(parameters):
    dataframe = pd.read_csv(parameters["dataframe"], sep='\t', header=0)
    train_dataframe, val_dataframe = sklearn.model_selection.train_test_split(dataframe, test_size=parameters['validation_split'],
                                                       stratify=dataframe["target"])
    train_dataframe.to_csv(parameters["output_train"], sep='\t', index=False)
    val_dataframe.to_csv(parameters["output_val"], sep='\t', index=False)
    return

def multiple_split_dataframe(list_of_dictionaries):
    for parameters in list_of_dictionaries:
        split_Dataframe(parameters)
    return

