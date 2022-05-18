import tensorflow as tf
from tensorflow.keras.layers import Dense
import pandas as pd

def clinical_model(input_num,output_num):
    #defines the model for predicting using the clinical data
    model = tf.keras.Sequential()
    model.add(Dense(8, input_dim=input_num, activation="relu"))
    model.add(Dense(output_num, activation="relu"))
    return model

def load_clinical_data(parameters):
    #loads the clinical data
    #parameters is a dictionary containing the following keys:
    #   - data_path: the path to the data
    #   - data_file: the name of the file containing the data
    #   - data_type: the type of data (clinical or clinical_test)
    #   - data_num: the number of data points
    #   - data_dim: the dimension of the data
    #   - data_labels: the labels for the data
    #   - data_labels_num: the number of labels
    #   - data_labels_dim: the dimension of the labels
    #   - data_labels_type: the type of labels (binary or categorical)
    #   - data_labels_names: the names of the labels
    #   - data_labels_names_num: the number of names of the labels
    #   - data_labels_names_dim: the dimension of the names of the labels
    #   - data_labels_names_type: the type of names of the labels (binary or categorical)

    #loads the data
    #~concatenates the continuos and the categorical data lists
    columns = parameters["continuos"] + parameters["categorical"]
    df = pd.read_csv(parameters["path"], names=columns)

    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[continuous])
    testContinuous = cs.transform(test[continuous])