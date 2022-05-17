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
    text_file = open(file, "w")

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
    text_file = open(file, "w")
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

