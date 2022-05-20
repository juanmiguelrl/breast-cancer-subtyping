import tensorflow as tf
from tensorflow.python.keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, LSTM
import numpy as np

def clinical_model(input_num,output_num):
    #defines the model for predicting using the clinical data
    model = tf.keras.Sequential()
    model.add(Dense(input_num, input_dim=input_num, activation="relu"))
    model.add(Dense(output_num, activation="softmax"))
    return model


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
##############################


class DataGenerator(tf.keras.utils.Sequence):
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
        X = list_IDs_temp

        return X#, y


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, x_col, y_col=None, batch_size=32, num_classes=None, shuffle=True):
        self.batch_size = batch_size
        self.df = dataframe
        self.indices = self.df.index.tolist()
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.x_col = x_col
        self.y_col = y_col
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]

        #X, y = self.__get_data(batch)
        X = self.__get_data(batch)
        return X#, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        #X =  self.df.loc[batch, self.x_col].values
        #X = self.df.loc[1].values
        X = self.df[1]
            #y =  # logic

            # for i, id in enumerate(batch):
            #     X[i,] =  # logic
            #     y[i] =  # labels

        return X#, y

from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras.utils import data_utils

class DataGenerator(data_utils.Sequence):
    def __init__(self, dataframe, x_col=None, y_col=None, batch_size=32, num_classes=None, shuffle=True):
        self.batch_size = batch_size
        self.df = dataframe
        self.indices = self.df.index.tolist()
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.x_col = x_col
        self.y_col = y_col
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]

        #X, y = self.__get_data(batch)
        X = self.__get_data(batch)
        return X

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        X =  self.df.loc[batch].values
        X = np.array([1,2,3,4,5])
        #X = self.df.loc[1].values
        #X = self.df[1]
            #y =  # logic

            #for i, id in enumerate(batch):
                #X[i,] =  # logic
            #     y[i] =  # labels

        return X#, y






from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras.utils import data_utils

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, dataframe, dfy, batch_size=32, shuffle=True,mode='train'):
        self.batch_size = batch_size
        self.df = dataframe
        self.indices = self.df.index.tolist()
        #self.num_classes = num_classes
        self.shuffle = shuffle
        #self.x_col = x_col
        #self.y_col = y_col
        self.dfy = dfy
        self.on_epoch_end()
        self.mode = mode

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]

        X, y = self.__get_data(batch)
        #X = self.__get_data(batch)
        return X,y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        X =  self.df.loc[batch].values
        y = self.dfy.loc[batch].values
        #X = self.df.loc[1].values
        #X = self.df[1]
            #y =  # logic

            #for i, id in enumerate(batch):
                #X[i,] =  # logic
            #     y[i] =  # labels

        return X, y




######################
dataframe = pd.read_csv("D:\\Escritorio\\tfg\\definitive_test\\classified.csv",sep="\t")
from sklearn.model_selection import train_test_split
train_dataframe, test_dataframe = train_test_split(dataframe, test_size=0.25, stratify=dataframe["target"])
##########
parameters = {"continuos" : ["age_at_initial_pathologic_diagnosis","size"], "categorical" : ["BRCA_Pathology"]}
cs = MinMaxScaler()
data = dataframe.copy()
data.set_index("filepath",inplace=True)
#data = data[parameters["continuos"] + parameters["categorical"] + ["target"]]
data = data[parameters["continuos"] + parameters["categorical"]]
data[parameters["continuos"]] = cs.fit_transform(data[parameters["continuos"]])
data = pd.get_dummies(data, columns=parameters["categorical"])

target = dataframe.copy()
target.set_index("filepath",inplace=True)
target = target["target"]
target = pd.get_dummies(target, columns=["target"])
#############

#datagen = DataGenerator(train_dataframe)
#datagen = DataGenerator(train_dataframe["BRCA_Pathology"], "BRCA_Pathology",dataframe, batch_size=32)
datagen = DataGenerator(data,target)

model = clinical_model(data.shape[1],target.shape[1])
#model = clinical_model_v2(train_Data.shape[1],train_y.shape[1])
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(datagen, epochs=100,
          steps_per_epoch=8,
          batch_size=8,
          verbose=1)


data.loc[["D:/Escritorio/tfg/test/images_together_filtered2\TCGA-A7-A0DA-01A-03-TS3.fec083e6-27fd-41ee-b44b-7bb1f9ec2d12.png","D:/Escritorio/tfg/test/images_together_filtered2\TCGA-BH-A0BD-01A-01-TSA.b92fbb4e-3f22-48d5-9b1e-af972a45232e.png"]].values
c.loc[["fec083e6-27fd-41ee-b44b-7bb1f9ec2d12","caa920cc-a74f-41a3-9b12-d8143dda8786"],["BRCA_Pathology"]]

#################################
class JoinedGen(tf.keras.utils.Sequence):
    def __init__(self, input_gen1, input_gen2, target_gen):
        self.gen1 = input_gen1
        self.gen2 = input_gen2
        self.gen3 = target_gen

        assert len(input_gen1) == len(input_gen2) == len(target_gen)

    def __len__(self):
        return len(self.gen1)

    def __getitem__(self, i):
        x1 = self.gen1[i]
        x2 = self.gen2[i]
        y = self.gen3[i]

        return [x1, x2], y

    def on_epoch_end(self):
        self.gen1.on_epoch_end()
        self.gen2.on_epoch_end()
        self.gen3.on_epoch_end()
        self.gen2.index_array = self.gen1.index_array
        self.gen3.index_array = self.gen1.index_array