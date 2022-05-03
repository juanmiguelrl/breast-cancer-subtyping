import argparse
import image_preprocess, store_images, filters, clasify, ann2, eval
import json
from datetime import datetime

def json_params(json_file):
    with open(json_file) as f:
        params = json.load(f)
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument("--d", help="downscale wsi images", required=False, default=False, action="store_true")
    parser.add_argument("--s", help="store images together", required=False, default=False, action="store_true")
    parser.add_argument("--f", help="filter the images", required=False, default=False, action="store_true")
    parser.add_argument("--c", help="clasify the images and split them in train and test sets", required=False,
                        default=False, action="store_true")
    parser.add_argument("--t", help="trains the network", required=False, default=False, action="store_true")
    parser.add_argument("--e", help="evaluate the model", required=False, default=False, action="store_true")
    parser.add_argument("--j", help="json file with the variables", required=True)

    args = parser.parse_args()
    print(args)

    PARAMS = json_params(args.j)

    if args.d:
        print("dowsncaling...")
        image_preprocess.downscale_from_manifest(PARAMS["downscale"]["manifest_path"], PARAMS["downscale"]["svsdirectory"], PARAMS["downscale"]["outputDirectory"], PARAMS["downscale"]["scale"],PARAMS["downscale"]["windows"])
    if args.s:
        print("resizing, cropping and storing images together...")
        store_images.store_images_together(PARAMS["store"]["destination_path"], PARAMS["store"]["input_dir"])
    if args.f:
        print("filtering images...")
        filters.filter_images(PARAMS["filter"]["input_dir"], PARAMS["filter"]["destination_path"], PARAMS["filter"]["prefix"], PARAMS["filter"]["resize_size"])
    if args.c:
        print("clasifying images...")
        clasify.clasify_images(PARAMS["clasify"]["input"], PARAMS["clasify"]["imgdir"], PARAMS["clasify"]["sourceDir"], PARAMS["clasify"]["newDirTOsplitImages"])

    if args.t or args.e:
        # Sets up a timestamped log directory.
        log_dir = PARAMS["logdir"] + datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.t:
        print("training network...")
        ann2.train_ann(PARAMS["ann"]["trainDir"], PARAMS["ann"]["testDir"], PARAMS["logdir"], PARAMS["ann"]["batch_size"], PARAMS["ann"]["epochs"],
                       PARAMS["n_gpus"], PARAMS["model_dir"],PARAMS["ann"]["learning_rate"],log_dir)
    if args.e:
        print("evaluating network...")
        eval.evaluate_ann(PARAMS["model_dir"],PARAMS["eval"]["testDir"],PARAMS["eval"]["batch_size"],log_dir)
