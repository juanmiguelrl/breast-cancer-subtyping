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
    parser.add_argument("--c", help="clasifies the images and slits them in train and test sets ", required=False,
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
        image_preprocess.downscale_from_manifest(PARAMS["downscale"]["manifest_path"], PARAMS["downscale"]["svsdirectory"], PARAMS["downscale"]["outputDirectory"], PARAMS["downscale"]["scale"],PARAMS["downscale"]["windows"])
    if args.s:
        print("storing images together...")
        store_images.store_images_together(PARAMS["store"]["destination_path"], PARAMS["store"]["input_dir"])
    if args.f:
        #default values for the filter
        filter = {"resize_size": (896,896) ,"only_tissue":True,"canny":False,"discard":True,"crop":True,
                  "resize":True,"remove_blue_pen":False,"remove_red_pen":False,"remove_green_pen":False,"only_one_tissue":True,"empty_threshold":0.5,
                  "canny_params":{"sigma":1.0,"low_threshold":0,"high_threshold":25}}
        filter.update(PARAMS["filter"])
        print("filtering images...")
        filters.filter_images(filter["input_dir"], filter["destination_path"], filter["resize_size"],
                              filter["only_tissue"],filter["canny"],filter["discard"],filter["crop"],
                              filter["resize"],filter["remove_blue_pen"],filter["remove_red_pen"],filter["remove_green_pen"],
                              filter["only_one_tissue"],filter["empty_threshold"],filter["canny_params"])
    if args.c:
        print("clasifying images...")
        clasify_img = {"simplify_stage": False}
        clasify_img.update(PARAMS["clasify"])
        clasify.clasify_images(clasify_img["input"], clasify_img["imgdir"], clasify_img["classification"], clasify_img["output_file"],clasify_img["simplify_stage"])
        #clasify.clasify_images_oldv2(PARAMS["clasify"]["input"], PARAMS["clasify"]["imgdir"], PARAMS["clasify"]["sourceDir"], PARAMS["clasify"]["newDirTOsplitImages"],PARAMS["classification"])

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
