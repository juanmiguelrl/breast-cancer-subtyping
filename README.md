# breast-cancer-subtyping

First, the data needs to be downloaded with the GDC client  
The GDC client module is activated in cesga with:  
module load gcccore/6.4.0 gdc-client/1.3.0-python-2.7.15  

After the download, all that has to be done is prepare a json file with the parameters needed at main.py
(json file explanation uncompleted at the moment)  
## json

### Manifest download options in json (--m):  
Compulsory:  
output_file : the path where the manifest will be stored  
Optional:  
projects : list with the projects to search for the query (has the project TCGA-BRCA as default)  
name_restrictions : List with restrictions for the name of the slides searched (for example ["\*TS\*"] will search for the files which contains TS in their names)  
files_data_format : the data format which is wanted for the files (svs as default)  
experimental_strategy : the experimental strategy of the files  
endpoint : the endpoint where the query request will be sent, it should not be necessary to modify it (as default it has https://api.gdc.cancer.gov/files)  

### GDC data download options in json (--cl):  
Compulsory:  
output_file : the path where the output will be stored  
manifest_path : the path where the manifest file is stored  
Optional:  
projects : list with the projects to search for the query (has the project TCGA-BRCA as default)  
name_restrictions : List with restrictions for the name of the slides searched (for example ["\*TS\*"] will search for the files which contains TS in their names)  
files_data_format : the data format which is wanted for the files (svs as default)   
experimental_strategy : the experimental strategy of the files  
expand : expand indicated to the API (as default has ["diagnoses","samples","files"])   
fields_dictionary : dictionary with the fields to download and the how are wanted to be named in the final output file  
(Avaliable fields: https://docs.gdc.cancer.gov/API/Users_Guide/Appendix_A_Available_Fields/)  
endpoint : the endpoint where the query request will be sent, it should not be necessary to modify it (as default it has https://api.gdc.cancer.gov/cases)  

Info about the GDC API at: https://docs.gdc.cancer.gov/API/Users_Guide/Getting_Started/ and notebook with GDC API examples at https://github.com/NCI-GDC/gdc-docs/blob/develop/Notebooks/API_April_2021/Webinar_April_2021.ipynb

### R data download options in json (--dr) (Requires R to be installed):
(The script script.R in the R folder can be used to download the PAM50 data for the BRCA data, this option can be used also to run other R scripts provided by the user)  
Compulsory:  
&emsp;executable : Boolean indicating if call Rscript or call a personalised path  
&emsp;r_script_path : path for the R script to use  
Optional:  
&emsp;r_path : personalised path to call R  
&emsp;arguments : arguments to pass to the R script  

### join dataframes options in json (--jd):
Compulsory:  
join_data : List of dictionaries containing each one the two dataframes that want to be joined, these dictionaries contain :  
&emsp; df1_path : Path of the first dataframe to join  
&emsp; df2 path : Path of the second dataframe to join  
&emsp; sep1 : The separator used in the firs dataframe  
&emsp; sep2 : The separator used in the second dataframe  
&emsp; join type : The join type which will be used (it uses pandas.Dataframe.merge, so the same joins options are available: ‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’)  
&emsp; left_on : The column name which will be used to join on in the first dataframe  
&emsp; right_on : The column name which will be used to join on in the second dataframe 
&emsp; output_file : The path for storing the resulting dataframe

### Slides download options in json (--dl) (it requires the gdc-downloader program): 
Compulsory:  
&emsp;manifest_file : The path to the manifest file  
&emsp;output_dir : The path where the slides will be downloaded  
Optional:  
&emsp;executable : To indicate if the gdc-client is an executable program in a path to indicate  
&emsp;executable_path_file : The path where the gdc-client is  
&emsp;command_for_gdc_client : The command for the gdc-client in case it can be called directly

### Preprocess filter options in json (--f):  
Compulsory:  
&emsp;input_dir : directory path with the images to preprocess (written in the format "directory\\*")  
&emsp;destination_path : directory path to store the preprocessed images  
Optional:  
&emsp;resize: bool to resize the image or not  
&emsp;resize_size : the new size of the image after being resized if the resize bool is set to true  
&emsp;only_tissue : bool if it is wanted to preprocess the image but without eliminating the non tissue part or not  
&emsp;canny : bool to apply canny filter to accentuate the slide structures or not  
&emsp;canny params : sigma,low_threshold,high_threshold  
&emsp;discard : bool to choose if discard images without enough tissue after being preprocessed or not  
&emsp;empty_threshold : the threshold that defines how much of the image has to be empty to be discarded, between 0 and 1  
&emsp;crop : bool to indicate if crop or not the image to adjust to the tissue size (it crops the image so all the blank &emsp;unnecessary space is removed reducing the image size)  
&emsp;remove_blue_pen : bool to indicate if remove the blue pen marks in the slides or not  
&emsp;remove_red_pen : bool to indicate if remove the red pen marks in the slides or not  
&emsp;remove_green_pen : bool to indicate if remove the green pen marks in the slides or not  
&emsp;only_one_tissue : bool to indicate if leave only the biggest tissue connected component or leave all  

### Clasify images from directory folders to dataframe (--cf): 
(This option stores in a new dataframe the images paths with a "target" column with the classification which the folders containing them have as name)  
List of dictionaries containing each one:  
&emsp;imgdir: Path to the directory which contains the folders of each class of images  
&emsp;extension: The extension of the images (to don`t get other type of files into the dataframe)  
&emsp;output_file: Path where the resulting dataframe will be stored  

### Clasify options in json (--c): 
(This option stores in a new dataframe the images paths with a "target" column with the classification to be used)  
List of dictionaries containing each one:  
Compulsory:  
&emsp;input: Path to the dataframe with the data to use in the classification  
&emsp;imgdir: Path to the folder containing the images to clasify  
&emsp;classification: Column name of the input dataframe which will be used to clasify  
&emsp;output_file: Path where the resulting dataframe will be stored  
Optional:  
&emsp;simplify_stage: Boolean if true it will create a new column in the output dataset with the column stage simplified (to 1,2,3,4)  

### Modify target columns options (--mt)  
This option modifies the target columns from the dataframe indicated.  
It is a list of dictionaries containing each one:  
Compulsory:  
&emsp;input : Path to the dataframe to modify   
&emsp;output : Path where the resulting dataframe will be stored  
&emsp;drop : List containing the values of the target that are wanted to be removed before any other change  
&emsp;new_group : Boolean, if true it will create new values in the target column for the values indicated  
&emsp;Compulsory if new_group is set to true:  
&emsp;&emsp; new_group_list : Dictionary containing lists with the values that are wanted to be renamed (and grouped together if there is more than one element in the list)  
&emsp;Compulsory if new_group or remove other is set to true:  
&emsp;&emsp;other_name : The name that will be used to rename all the values which are not in the new_group list (if new_group is set to true)  
&emsp; remove_other : Boolean, if true it will remove all the rows with the value specified in other_name in the target column (this is executed after new_group)  

### Split dataframe (--sd): 
(This option splits the dataframes provided into 2 dataframes with the same ratio of each class than the original)  
List of dictionaries containing each one:  
&emsp;dataframe: Path to the dataframe to be splitted.  
&emsp;validation_split: Number between 0 and 1 which indicates which percentage of the dataframe goes to each new dataframe (0.4 for example will result in the first dataframe having 60% of the data and the second one having a 40%)   
&emsp;target: The column of the dataframe with the classes that will be used to stratify (divides in the same ratio)  
&emsp;output_train: Path where the first resulting dataframe will be stored  
&emsp;output_val: Path where the second resulting dataframe will be stored  
(being a list allows to for example make a first split into 60% for training and 40% for other dataframe that in the next element of the list can be splitted 50% to have a train dataframe with 20% of the original data and a test dataframe with another 20% of the original data)


### Train options in json (--t): (not finished yet) 
Compulsory:  
&emsp;balance_data : boolean, if true it balance the data taking randomly the same amount of samples of each clasify option

## Program options:  
--j : followed by the json file path with the program configuration (needed for the different options of the program to work correctly)  
(Options described in order, all of them need to be executed at least once in order to be able to use the next one)  
&emsp;--m : download manifest from GDC  
&emsp;--cl : download additional data from GDC  
&emsp;--dr : run a R script to download data or any other R script wanted by the user
&emsp;--dl : download slides from a manifest file using the gdc-client program  
&emsp;--d : to downscale the wsi files  
&emsp;--s : to store the images together  
&emsp;--f : filters the images applying the options indicated  
&emsp;--c : clasifies the images and slits them in train and test sets  
&emsp;--t : this option os to execute the training of the neural network  
&emsp;--e : to evaluate the model indicated with the dataset indicated  
(These next options are necessary for the training option):  
&emsp;--epochs : to indicate the number of epochs that will be done at the training  
&emsp;--batch _size: to indicate the batch size at the training  
&emsp;--n_gpus: to indicate how many gpus will be used during the training  
(These next options are optional for the training option):  
&emsp;--l : to log during the training or not  
&emsp;--n or --nni : to use nni or not during the training

