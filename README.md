# breast-cancer-subtyping

First, the data needs to be downloaded with the GDC client  
The GDC client module is activated in cesga with:  
module load gcccore/6.4.0 gdc-client/1.3.0-python-2.7.15  

After the download, all that has to be done is prepare a json file with the parameters needed at main.py
(json file example soon)  

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
