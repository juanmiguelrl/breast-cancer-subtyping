import argparse
import image_preprocess, store_images, filters, clasify, ann2, eval, download_manifest
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
    parser.add_argument("--d", help="downscale wsi images", required=False, default=False, action="store_true")
    parser.add_argument("--s", help="store images together", required=False, default=False, action="store_true")
    parser.add_argument("--f", help="filter the images", required=False, default=False, action="store_true")
    parser.add_argument("--c", help="clasify the images and split them in train and test sets", required=False,
                        default=False, action="store_true")
    parser.add_argument("--t", help="trains the network", required=False, default=False, action="store_true")
    parser.add_argument("--e", help="evaluate the model", required=False, default=False, action="store_true")
    parser.add_argument("--j", help="json file with the variables", required=True)
    parser.add_argument("--l", help="to log during the training or not", required=False, default=False, action="store_true")
    parser.add_argument("--n","--nni", help="to use nni or not", required=False, default=False, action="store_true")

    args = parser.parse_args()
    print(args)

    PARAMS = json_params(args.j)

    if args.n:
        PARAMS = nni_params(PARAMS)

    if args.m:
        #default values for the manifest
        manifest = {"projects": ["TCGA-BRCA"], "name_restrictions": ["*"],"endpoint": 'https://api.gdc.cancer.gov/files'}
        manifest.update(PARAMS["manifest"])
        download_manifest.download_manifest_from_GDC(manifest["output_file"],manifest["projects"],manifest["name_restrictions"],manifest["endpoint"])
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
        clasify.clasify_images(PARAMS["clasify"]["input"], PARAMS["clasify"]["imgdir"], PARAMS["clasify"]["sourceDir"], PARAMS["clasify"]["newDirTOsplitImages"],PARAMS["classification"])

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
