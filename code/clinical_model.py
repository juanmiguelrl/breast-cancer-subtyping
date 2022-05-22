import tensorflow as tf
from tensorflow.python.keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, LSTM
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Dense

def clinical_model(input_num,output_num):
    #defines the model for predicting using the clinical data
    # model = Sequential()
    # tf.keras.layers.Input(shape=(input_num,))
    # model.add(Dense(input_num, input_dim=input_num, activation="relu"))
    # model.add(Dense(output_num, activation="softmax"))

    print(input_num)
    input = Input(shape=(input_num,))
    model = Dense(input_num, activation="relu")(input)
    model = Dense(output_num, activation="softmax") (model)
    model = Model(inputs=input, outputs=model)

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
#
# dataframe = pd.read_csv("D:\\Escritorio\\tfg\\definitive_test\\classified.csv",sep="\t")
# from sklearn.model_selection import train_test_split
# train_dataframe, test_dataframe = train_test_split(dataframe, test_size=0.25, stratify=dataframe["target"])
# parameters = {"continuos" : ["age_at_initial_pathologic_diagnosis"], "categorical" : ["BRCA_Pathology"]}
# train_Data = load_clinical_data(parameters,train_dataframe,dataframe)
# test_data = load_clinical_data(parameters,test_dataframe,dataframe)
#
# # parameters2 = {"continuos" : ["age_at_initial_pathologic_diagnosis"], "categorical" : ["BRCA_Subtype_PAM50"]}
# # train_Data2,test_data2 = load_clinical_data_old(parameters2,train_dataframe,test_dataframe,dataframe)
#
#
#
# #one hot encoding for the value to predict
# zipBinarizer = LabelBinarizer().fit(dataframe["target"])
# train_y = zipBinarizer.transform(train_dataframe["target"])
# test_y = zipBinarizer.transform(test_dataframe["target"])
#
#
# model = clinical_model(train_Data.shape[1],train_y.shape[1])
# #model = clinical_model_v2(train_Data.shape[1],train_y.shape[1])
# model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
#
# model.fit(
# 	x=train_Data, y=train_y,
# 	validation_data=(test_data, test_y),
# 	epochs=100, batch_size=8)


# model = clinical_model(train_Data2.shape[1],5)
# model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
#
# model.fit(
# 	x=train_Data2, y=train_y,
# 	validation_data=(test_data2, test_y),
# 	epochs=200, batch_size=8)










##############################
from tensorflow.keras.utils import Sequence
import math

class ClinicalDataGenerator(Sequence):
    def __init__(self, dataframe, dfy, batch_size=32, shuffle=False,mode='train'):
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
        #return len(self.indices) // self.batch_size
        return math.ceil(len(self.indices) / self.batch_size)
    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indices of the batch
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        batch = [self.indices[k] for k in index]
        # Generate data
        X, y = self.__get_data(batch)
        return X, y

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
# dataframe = pd.read_csv("D:\\Escritorio\\tfg\\definitive_test\\classified.csv",sep="\t")
# from sklearn.model_selection import train_test_split
# train_dataframe, test_dataframe = train_test_split(dataframe, test_size=0.25, stratify=dataframe["target"])
# ##########
# parameters = {"continuos" : ["age_at_initial_pathologic_diagnosis","size"], "categorical" : ["BRCA_Pathology"]}
# cs = MinMaxScaler()
# data = dataframe.copy()
# data.set_index("filepath",inplace=True)
# #data = data[parameters["continuos"] + parameters["categorical"] + ["target"]]
# data = data[parameters["continuos"] + parameters["categorical"]]
# data[parameters["continuos"]] = cs.fit_transform(data[parameters["continuos"]])
# data = pd.get_dummies(data, columns=parameters["categorical"])
#
# target = dataframe.copy()
# target.set_index("filepath",inplace=True)
# target = target["target"]
# target = pd.get_dummies(target, columns=["target"])
#############

#process the data for the model which uses the clinical data
def process_clinical_data(dataframe,parameters):
    cs = MinMaxScaler()
    data = dataframe.copy()
    #data.set_index("filepath", inplace=True)
    # data = data[parameters["continuos"] + parameters["categorical"] + ["target"]]
    data = data[parameters["continuos"] + parameters["categorical"]]
    data[parameters["continuos"]] = cs.fit_transform(data[parameters["continuos"]])
    data = pd.get_dummies(data, columns=parameters["categorical"])

    target = dataframe.copy()
    #target.set_index("filepath", inplace=True)
    target = target["target"]
    target = pd.get_dummies(target, columns=["target"])

    return data,target

#
# dataframe = pd.read_csv("D:\\Escritorio\\tfg\\definitive_test\\classified.csv",sep="\t")
# from sklearn.model_selection import train_test_split
# train_dataframe, test_dataframe = train_test_split(dataframe, test_size=0.25, stratify=dataframe["target"])
# #train_dataframe.set_index("filepath", inplace=True)
# parameters = {"continuos" : ["age_at_initial_pathologic_diagnosis","size"], "categorical" : ["BRCA_Pathology"]}
# data,target = process_clinical_data(dataframe,parameters)
#
# #datagen = DataGenerator(train_dataframe)
# #datagen = DataGenerator(train_dataframe["BRCA_Pathology"], "BRCA_Pathology",dataframe, batch_size=32)
# datagen = DataGenerator(data,target)
#
# model = clinical_model(data.shape[1],target.shape[1])
# #model = clinical_model_v2(train_Data.shape[1],train_y.shape[1])
# model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
# # model.fit(datagen, epochs=100,
# #           steps_per_epoch=8,
# #           batch_size=8,
# #           verbose=1)
# #
# # model.predict(datagen[1][0])

#################################
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# preprocess_func = tf.keras.applications.vgg16.preprocess_input
#
# parameters = {"x_col" : "filepath", "y_col" : "target",'target_size': (224,224),'preprocess_func' : preprocess_func,"batch_size" : 32,
#               "save_to_dir_train" : "D:\\Escritorio\\tfg\\definitive_test\\generator_images\\train",
#               "save_to_dir_validation" : "D:\\Escritorio\\tfg\\definitive_test\\generator_images\\validation"}
#
# train_datagen = ImageDataGenerator(preprocessing_function=preprocess_func,
#                                    rescale=1. / 255)
#
# validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_func,
#                                         rescale=1. / 255)
#
# train_generator = train_datagen.flow_from_dataframe(
#     dataframe,
#     x_col=parameters['x_col'],
#     y_col=parameters['y_col'],
#     target_size=parameters['target_size'],
#     batch_size=parameters['batch_size'],
#     class_mode='categorical',
#     shuffle=False,
#     #save_to_dir=parameters['save_to_dir_train']  # ,
#     # shuffle=True
# )
#
# validation_generator = validation_datagen.flow_from_dataframe(
#     test_dataframe,
#     x_col=parameters['x_col'],
#     y_col=parameters['y_col'],
#     target_size=parameters['target_size'],
#     batch_size=parameters['batch_size'],
#     class_mode='categorical',
#     shuffle=False,
#     #save_to_dir=parameters['save_to_dir_validation']
# )

#################################
class JoinedGen(tf.keras.utils.Sequence):
    def __init__(self, input_gen1, input_gen2,shuffle=True#, target_gen
    ):
        self.gen1 = input_gen1
        self.gen2 = input_gen2
        self.shuffle = shuffle
        #self.gen3 = target_gen

        print(self.gen1.__len__())
        print(self.gen2.__len__())

        assert len(input_gen1) == len(input_gen2)
        # == len(target_gen)
        #self.gen2.indices = self.gen1.index_array
        # if self.shuffle:
        #     np.random.shuffle(self.gen1.index_array)
        #     self.gen2.indices = self.gen1.index_array

    def __len__(self):
        return len(self.gen1)

    def __getitem__(self, i):
        x1 = self.gen1[i]
        x2 = self.gen2[i]
        #y = self.gen3[i]
        # print(self.gen1.index_array)
        # print("\n\n\n")
        # print(self.gen2.indices)
        return [x1[0], x2[0]], x1[1]

    def on_epoch_end(self):
        self.gen1.on_epoch_end()
        self.gen2.on_epoch_end()
        #self.gen3.on_epoch_end()
        #if self.shuffle == True:
        # print(self.gen1.index_array)
        # print("\n\n\n")
        # print(self.gen2.indices)
        if self.shuffle:
            np.random.shuffle(self.gen1.index_array)
            self.gen2.indices = self.gen1.index_array
        # print("\n\n\n")
        # print(max(self.gen1.index_array))
        # print("\n\n\n")
        # print(max(self.gen2.indices))
        #self.gen3.index_array = self.gen1.index_array

#################################
# join = JoinedGen(train_generator, datagen)
#
# ff = join[1]
#
# join.on_epoch_end()