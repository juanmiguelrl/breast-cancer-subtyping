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
##############################
from tensorflow.keras.utils import Sequence
import math

def clinical_model(input_num,output_num,depth,width,activation_function):
    #defines the model for predicting using the clinical data

    print(input_num)
    input = Input(shape=(input_num,))
    model = Dense(input_num * width, activation=activation_function)(input)
    for i in range(1,depth):
        model = Dense(input_num*width, activation=activation_function)(model)
    model = Dense(output_num, activation="softmax")(model)
    model = Model(inputs=input, outputs=model)

    return model




class ClinicalDataGenerator(Sequence):
    def __init__(self, dataframe, dfy, batch_size=32, shuffle=False,mode='train'):
        self.batch_size = batch_size
        self.data = dataframe
        self.indices = self.data.index.tolist()
        #self.num_classes = num_classes
        self.shuffle = shuffle
        #self.x_col = x_col
        #self.y_col = y_col
        self.datay = dfy
        self.on_epoch_end()
        self.mode = mode
        # dff = pd.DataFrame()
        # dff["t"] = pd.get_dummies(dfy).idxmax(1)
        # self.classes = dff.groupby("t").ngroup().to_numpy()
        # self.class_indices = class_indices
        # self.classes = dfy.map(dict)

        def generate_classes(df):
            target = pd.DataFrame()
            target["t"] = pd.get_dummies(df).idxmax(1)
            l = []
            for element in target["t"].unique():
                l.append(element)
            l = sorted(l)
            i = 0
            dict = {}
            for element in l:
                # add to dictionary with i
                dict[element] = i
                i += 1
            target["t"] = target["t"].map(dict)
            print(target["t"])
            print(dict)
            return target["t"].to_numpy(), dict

        self.classes, self.class_indices = generate_classes(dfy) #pd.factorize(self.classes)



    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indices of the batch
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        batch = [self.indices[k] for k in index]
        # Generate data
        X, y = self.__get_data(batch)
        # print(X)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        # print("******************")
        # print(batch)
        # print("******************")
        # X =  self.df.loc[batch].values
        # y = self.dfy.loc[batch].values
        X = self.data.reindex(batch).values
        y = self.datay.reindex(batch).values

        # print(self.data.index.values.tolist())

        return X, y


#process the data for the model which uses the clinical data
def process_clinical_data(dataframe,parameters):
    cs = MinMaxScaler()
    data = dataframe.copy()
    #data.set_index("filepath", inplace=True)
    # data = data[parameters["continuos"] + parameters["categorical"] + ["target"]]
    for column in data:
        if column in parameters["continuos"]:
            data[column].fillna(0, inplace=True)
        if column in parameters["categorical"]:
            data[column].replace(np.nan, "unknown")
    data = data[parameters["continuos"] + parameters["categorical"] + ["target","filepath"]]
    data[parameters["continuos"]] = cs.fit_transform(data[parameters["continuos"]])
    data = pd.get_dummies(data, columns=parameters["categorical"])

    target = dataframe.copy()
    target = target["target"]
    target = pd.get_dummies(target, columns=["target"])

    return data,target


class JoinedGen(tf.keras.utils.Sequence):
    def __init__(self, input_gen1, input_gen2,shuffle=True#, target_gen
    ):
        self.gen1 = input_gen1
        self.gen2 = input_gen2
        self.shuffle = shuffle
        self.classes = self.gen1.classes
        self.class_indices = self.gen1.class_indices

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
        # print(self.gen1.index_array)
        # print("\n\n\n")
        # print(self.gen2.indices)

        #print each row of x1,x2
        # print("\n")
        # for i in range(0, len(x1[0])):
        #     #print(x1[0][i])
        #     print(x2[0][i])
        #     print(x1[1][i])
        # print("\n")

        return [x1[0], x2[0]], x1[1]

    def on_epoch_end(self):
        self.gen1.on_epoch_end()
        #self.gen2.on_epoch_end()
        #if self.shuffle == True:
        # print(self.gen1.index_array)
        # print("\n\n\n")
        # print(self.gen2.indices)
        # if self.shuffle:
        #     np.random.shuffle(self.gen1.index_array)
        self.gen2.indices = self.gen1.index_array
        # print("\n\n\n")
        # print(max(self.gen1.index_array))
        # print("\n\n\n")
        # print(max(self.gen2.indices))