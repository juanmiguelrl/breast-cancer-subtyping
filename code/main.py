import argparse
import image_preprocess, store_images, filters, clasify, ann2, eval, util
import json
from datetime import datetime
import nni

def json_params(json_file):
    with open(json_file) as f:
        params = json.load(f)
    return params

def nni_params(params):
    print("uptdating params with nni")
    updated_params = nni.get_next_parameter()
    params["ann"].update(updated_params)
    print("Config after update with NNI config:")
    print(json.dumps(PARAMS, indent=2))
    return PARAMS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument("--m", help="download manifest from GDC", required=False, default=False, action="store_true")
    parser.add_argument("--cl", help="download clinical from GDC", required=False, default=False, action="store_true")
    parser.add_argument("--dl", help="download slides from a manifest file using the gdc-client program", required=False, default=False, action="store_true")
    parser.add_argument("--dr", help="download data using a R script",required=False, default=False, action="store_true")
    parser.add_argument("--jd", help="joins the dataframes indicated", required=False, default=False,action="store_true")
    parser.add_argument("--d", help="downscale wsi images", required=False, default=False, action="store_true")
    parser.add_argument("--s", help="store images together", required=False, default=False, action="store_true")
    parser.add_argument("--f", help="filters the images applying the options indicated", required=False, default=False, action="store_true")
    parser.add_argument("--mt", help="modifies the target column of the dataframes indicated", required=False, default=False,
                        action="store_true")
    parser.add_argument("--cf", help="clasifies the images from a directory into a dataframe", required=False,
                        default=False, action="store_true")
    parser.add_argument("--c", help="clasifies the images and splits them in train and test sets ", required=False,
                        default=False, action="store_true")
    parser.add_argument("--pc", help="process clinical dataframe)", required=False,
                        default=False, action="store_true")
    parser.add_argument("--sd", help="splits a dataframe (for example into train and validation)", required=False,
                        default=False, action="store_true")
    parser.add_argument("--t", help="execute the training of the neural network", required=False, default=False, action="store_true")
    parser.add_argument("--e", help="evaluate the model indicated with the dataset indicated", required=False, default=False, action="store_true")
    parser.add_argument("--j", help="json file with the variables", required=True)
    parser.add_argument("--l", help="to log during the training or not", required=False, default=False, action="store_true")
    parser.add_argument("--n","--nni", help="to use nni or not during the training", required=False, default=False, action="store_true")

    args = parser.parse_args()
    print(args)

    PARAMS = json_params(args.j)

    if args.n:
        PARAMS = nni_params(PARAMS)

    if args.m:
        #default values for the manifest
        #if it is wanted to not apply a filter its value has to be passed as ["*"] and the API will interpret it as any value is valid for that field
        manifest = {"projects": ["TCGA-BRCA"], "name_restrictions": ["*"],"files_data_format": ["svs"],
                    "experimental_strategy":["*"] ,"endpoint": 'https://api.gdc.cancer.gov/files'}
        manifest.update(PARAMS["manifest"])
        util.download_manifest_from_GDC(manifest["output_file"],manifest["projects"],manifest["name_restrictions"],manifest["files_data_format"],
                                                     manifest["experimental_strategy"],manifest["endpoint"])
    if args.cl:
        #default values for the clinical
        clinical = {"projects": ["TCGA-BRCA"], "name_restrictions": ["*"],"files_data_format": ["svs"],
                    "experimental_strategy":["*"] ,"expand":["diagnoses","samples","files"],
                    "fields_dictionary":{"stage":"diagnoses.0.ajcc_pathologic_stage","icd_10_code":"diagnoses.0.icd_10_code"},
                    "endpoint": 'https://api.gdc.cancer.gov/cases'}
        clinical.update(PARAMS["clinical"])
        util.download_data(clinical["output_file"],clinical["manifest_path"],clinical["projects"],clinical["name_restrictions"],
                                                     clinical["fields_dictionary"],clinical["experimental_strategy"],
                                                     clinical["expand"],clinical["files_data_format"],clinical["endpoint"])
    if args.dr:
        #default values for the r_donwload
        r_donwload = {"r_path": "","arguments":[]}
        r_donwload.update(PARAMS["r_donwload"])
        util.download_data_with_R(r_donwload["executable"],r_donwload["r_path"],r_donwload["r_script_path"],r_donwload["arguments"])
    if args.jd:
        #default values for the jd_donwload
        #join_data = {}
        #join_data.update(PARAMS["join_data"])
        util.join_data(PARAMS["join_data"])
    if args.dl:
        #default values for the slides
        slides = {"executable": False, "executable_path_file": None,"command_for_gdc_client" : "gdc-client"}
        slides.update(PARAMS["slides"])
        util.download_images_from_manifest(slides["manifest_file"],slides["output_dir"],slides["executable"],slides["executable_path_file"],slides["command_for_gdc_client"])

    if args.d:
        print("dowsncaling...")
        downscale = {"openslide_path": None,"manifest_path" : None,"store_together":True}
        downscale.update(PARAMS["downscale"])
        image_preprocess.downscale_from_manifest(downscale["manifest_path"], downscale["svsdirectory"], downscale["outputDirectory"], downscale["scale"],downscale["store_together"],downscale["openslide_path"])
    if args.s:
        print("storing images together...")
        store_images.store_images_together(PARAMS["store"]["destination_path"], PARAMS["store"]["input_dir"])
    if args.f:
        print("filtering images...")
        filters.filter_images_multiple(PARAMS["filter"])
    if args.cf:
        print("clasifying images...")
        clasify.dataframe_from_directory_multiple(PARAMS["dataframe_from_directory"])
    if args.c:
        print("clasifying images...")
        clasify.clasify_multiple(PARAMS["clasify"])
        #clasify.clasify_images_oldv2(PARAMS["clasify"]["input"], PARAMS["clasify"]["imgdir"], PARAMS["clasify"]["sourceDir"], PARAMS["clasify"]["newDirTOsplitImages"],PARAMS["classification"])
    if args.mt:
        print("modifying target column...")
        util.modify_multiple_targets(PARAMS["modify_target"])
    if args.pc:
        print("Processing clinical dataframe...")
        util.multiple_process_clinical_data(PARAMS["process_clinical_dataframe"])
    if args.sd:
        print("Splitting dataframe...")
        util.multiple_split_dataframe(PARAMS["split_dataframe"])
    if args.t or args.e:
        # Sets up a timestamped log directory.
        log_dir = PARAMS["logdir"] + datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.t:
        print("training network...")
        ann2.train_ann(PARAMS["ann"],PARAMS["model_dir"],log_dir,args.n)
        #ann3.train_ann(PARAMS["ann"]["trainDir"], PARAMS["ann"]["testDir"], PARAMS["logdir"], PARAMS["ann"]["batch_size"], PARAMS["ann"]["epochs"],
        #               PARAMS["n_gpus"], PARAMS["model_dir"],PARAMS["ann"]["learning_rate"],log_dir,args.l)
    if args.e:
        print("evaluating network...")
        eval.evaluate_ann(PARAMS["model_dir"],PARAMS["eval"]["testDir"],PARAMS["eval"]["batch_size"],log_dir)
