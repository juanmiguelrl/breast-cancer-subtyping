import tensorflow as tf
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, LSTM

def clinical_model(input_num,output_num):
    #defines the model for predicting using the clinical data
    model = tf.keras.Sequential()
    model.add(Dense(input_num, input_dim=input_num, activation="relu"))
    model.add(Dense(output_num, activation="softmax"))
    return model

def clinical_model_v2(input_num,output_num):
    #defines the model for predicting using the clinical data
    model = Sequential()
    model.add(Dense(128, input_dim=input_num, activation="relu"))
    model.add(LSTM(128, activation='relu',
                   input_shape=(1000, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    #model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(output_num, activation="relu"))

    # opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
    #
    # model.compile(optimizer='rmsprop',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])
    return model

# def load_clinical_data_old(parameters,train,test,df):
#     #loads the clinical data
#     #~concatenates the continuos and the categorical data lists
#     columns = parameters["continuos"] + parameters["categorical"]
#     #df = pd.read_csv(parameters["path"], names=columns)
#
#     # performin min-max scaling each continuous feature column to
#     # the range [0, 1]
#     cs = MinMaxScaler()
#     trainContinuous = cs.fit_transform(train[parameters["continuos"]])
#     testContinuous = cs.transform(test[parameters["continuos"]])
#
#     # one-hot encode the zip code categorical data (by definition of
#     # one-hot encoding, all output features are now in the range [0, 1])
#     zipBinarizer = LabelBinarizer().fit(df[parameters["categorical"]])
#     trainCategorical = zipBinarizer.transform(train[parameters["categorical"]])
#     testCategorical = zipBinarizer.transform(test[parameters["categorical"]])
#
#
#     # construct our training and testing data points by concatenating
#     # the categorical features with the continuous features
#     trainX = np.hstack([trainCategorical, trainContinuous])
#     testX = np.hstack([testCategorical, testContinuous])
#
#     return trainX,testX
#
#
# from tensorflow import feature_column
#
# def load_clinical_data(parameters,data,df):
#     # performin min-max scaling each continuous feature column to
#     # the range [0, 1]
#     cs = MinMaxScaler()
#     data = cs.fit_transform(data[parameters["continuos"]])
#     feature_columns = []
#     for header in parameters["continuos"]:
#         feature_columns.append(feature_column.numeric_column(header))
#
#     for col_name in parameters["categorical"]:
#         categorical_column = feature_column.categorical_column_with_vocabulary_list(
#             col_name, dataframe[col_name].unique())
#         indicator_column = feature_column.indicator_column(categorical_column)
#         feature_columns.append(indicator_column)
#
#     #loads the clinical data
#     #~concatenates the continuos and the categorical data lists
#     columns = parameters["continuos"] + parameters["categorical"]
#     #df = pd.read_csv(parameters["path"], names=columns)
#
#
#     # cs = MinMaxScaler()
#     # trainContinuous = cs.fit_transform(train[parameters["continuos"]])
#     # testContinuous = cs.transform(test[parameters["continuos"]])
#
#     # one-hot encode the zip code categorical data (by definition of
#     # one-hot encoding, all output features are now in the range [0, 1])
#     # zipBinarizer = LabelBinarizer().fit(df[parameters["categorical"]])
#     # trainCategorical = zipBinarizer.transform(train[parameters["categorical"]])
#     # testCategorical = zipBinarizer.transform(test[parameters["categorical"]])
#
#
#     # construct our training and testing data points by concatenating
#     # the categorical features with the continuous features
#     # trainX = np.hstack([trainCategorical, trainContinuous])
#     # testX = np.hstack([testCategorical, testContinuous])
#
#     return feature_columns


def load_clinical_data(parameters,data,df):
    #loads the clinical data
    #~concatenates the continuos and the categorical data lists
    columns = parameters["continuos"] + parameters["categorical"]
    #df = pd.read_csv(parameters["path"], names=columns)

    # performin min-max scaling each continuous feature column to
    # the range [0, 1]
    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(data[parameters["continuos"]])

    # one-hot encode the zip code categorical data (by definition of
    # one-hot encoding, all output features are now in the range [0, 1])
    # create empty dataframe
    #trainCategorical = pd.DataFrame()
    # for col_name in parameters["categorical"]:
    #     zipBinarizer = LabelBinarizer().fit(df[col_name])
    #     #trainCategorical = zipBinarizer.transform(data[col_name])
    #     np.hstack([trainCategorical,zipBinarizer.transform(data[col_name])])
    trainCategorical = data[parameters["categorical"]]
    trainCategorical = pd.get_dummies(trainCategorical, columns=parameters["categorical"])


    # construct our training and testing data points by concatenating
    # the categorical features with the continuous features
    trainX = np.hstack([trainCategorical, trainContinuous])

    return trainX


import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, #labels,
                 batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        #self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X#, y = self.__data_generation(list_IDs_temp)

        return X#, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            #y[i] = self.labels[ID]

        return X#, keras.utils.to_categorical(y, num_classes=self.n_classes)

dataframe = pd.read_csv("D:\\Escritorio\\tfg\\definitive_test\\classified.csv",sep="\t")
from sklearn.model_selection import train_test_split
train_dataframe, test_dataframe = train_test_split(dataframe, test_size=0.25, stratify=dataframe["target"])
parameters = {"continuos" : ["age_at_initial_pathologic_diagnosis"], "categorical" : ["BRCA_Pathology"]}
train_Data = load_clinical_data(parameters,train_dataframe,dataframe)
test_data = load_clinical_data(parameters,test_dataframe,dataframe)

# parameters2 = {"continuos" : ["age_at_initial_pathologic_diagnosis"], "categorical" : ["BRCA_Subtype_PAM50"]}
# train_Data2,test_data2 = load_clinical_data_old(parameters2,train_dataframe,test_dataframe,dataframe)



#one hot encoding for the value to predict
zipBinarizer = LabelBinarizer().fit(dataframe["target"])
train_y = zipBinarizer.transform(train_dataframe["target"])
test_y = zipBinarizer.transform(test_dataframe["target"])


model = clinical_model(train_Data.shape[1],train_y.shape[1])
#model = clinical_model_v2(train_Data.shape[1],train_y.shape[1])
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

model.fit(
	x=train_Data, y=train_y,
	validation_data=(test_data, test_y),
	epochs=100, batch_size=8)


# model = clinical_model(train_Data2.shape[1],5)
# model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
#
# model.fit(
# 	x=train_Data2, y=train_y,
# 	validation_data=(test_data2, test_y),
# 	epochs=200, batch_size=8)