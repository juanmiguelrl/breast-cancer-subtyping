# breast-cancer-subtyping

First, the data needs to be downloaded with the GDC client  
The GDC client module is activated in cesga with:  
module load gcccore/6.4.0 gdc-client/1.3.0-python-2.7.15  

After the download, all that has to be done is prepare a json file with the parameters needed at main.py
(json file explanation uncompleted at the moment)  

Preprocess options in json (filter):  
input_dir : directory path with the images to preprocess (written in the format "directory\\*")  
destination_path : directory path to store the preprocessed images  
resize: bool to resize the image or not  
resize_size : the new size of the image after being resized if the resize bool is set to true  
only_tissue : bool if it is wanted to preprocess the image but without eliminating the non tissue part or not  
canny : bool to apply canny filter to accentuate the slide structures or not  
canny params : sigma,low_threshold,high_threshold  
discard : bool to choose if discard images without enough tissue after being preprocessed or not  
empty_threshold : the threshold that defines how much of the image has to be empty to be discarded, between 0 and 1  
crop : bool to indicate if crop or not the image to adjust to the tissue size (it crops the image so all the blank unnecessary space is removed reducing the image size)  
remove_blue_pen : bool to indicate if remove the blue pen marks in the slides or not  
remove_red_pen : bool to indicate if remove the red pen marks in the slides or not  
remove_green_pen : bool to indicate if remove the green pen marks in the slides or not  
only_one_tissue : bool to indicate if leave only the biggest tissue connected component or leave all  


Program options:  
(Options described in order, all of them need to be executed at least once in order to be able to use the next one)  
--d : to downscale the wsi files  
--s : to store the images together  
--f : filters the images, crops them to take only one tissue sample and rescales them to the desired size  
--c : clasifies the images and slits them in train and test sets  
--t : this option os to execute the training of the neural network  
--e : to evaluate the model indicated with the dataset indicated  
(These next options are necessary for the training option):  
--epochs : to indicate the number of epochs that will be done at the training  
--batch _size: to indicate the batch size at the training  
--n_gpus: to indicate how many gpus will be used during the training  

