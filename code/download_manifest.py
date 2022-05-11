import requests
import json
import pandas as pd


def download_manifest_from_GDC(output_file,projects,name_restrictions,endpoint='https://api.gdc.cancer.gov/files'):
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
                    "value": "svs"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "files.experimental_strategy",
                    "value": "Tissue Slide"
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
        "size": "10000"
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