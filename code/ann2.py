import pandas as pd

from model import build_model
from clinical_model import process_clinical_data,ClinicalDataGenerator,JoinedGen
import numpy as np
import tensorflow as tf
import nni
import math
from tensorflow import keras

import sklearn.metrics
from eval import  plot_confusion_matrix,plot_to_image,confusion_matrix_callback,log_learning_rate_callback, log_nni_callback, confusion_matrix_test_callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from util import calculate_class_weights
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split

def train_ann( parameters,model_dir,log_dir,nni_activated):

    print(tf.config.list_physical_devices())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(device_lib.list_local_devices())

############################################
    #dataframe
    train_dataframe = pd.read_csv(parameters['train_dataframe'],sep="\t")
    valid_dataframe = pd.read_csv(parameters['val_dataframe'], sep="\t")
    test_dataframe = pd.read_csv(parameters['test_dataframe'], sep="\t")
    if parameters["balance_data"]:
        print("train before balance\n")
        print(train_dataframe["target"].value_counts())
        g = train_dataframe.groupby('target')
        train_dataframe = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)).reset_index(drop=True)
        print("train after balance\n")
        print(train_dataframe["target"].value_counts())

        print("validation before balance\n")
        print(valid_dataframe["target"].value_counts())
        g = valid_dataframe.groupby('target')
        valid_dataframe = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)).reset_index(drop=True)
        print("validation after balance\n")
        print(valid_dataframe["target"].value_counts())




############################################
    #prepare data generators with clinical data
    if parameters["clinical_model"]:
        # train_dataframe = process_clinical_data(train_dataframe,parameters["clinical_columns"])[0].reset_index(drop=True)
        # valid_dataframe = process_clinical_data(valid_dataframe,parameters["clinical_columns"])[0].reset_index(drop=True)
        # test_dataframe = process_clinical_data(test_dataframe, parameters["clinical_columns"])[0].reset_index(drop=True)

        clinical_train_target = pd.get_dummies(train_dataframe.copy()["target"], columns=["target"])
        clinical_valid_target = pd.get_dummies(valid_dataframe.copy()["target"], columns=["target"])
        clinical_test_target = pd.get_dummies(test_dataframe.copy()["target"], columns=["target"])
        clinical_train_dataframe = train_dataframe.copy().drop(columns=["target","filepath"])
        clinical_valid_dataframe = valid_dataframe.copy().drop(columns=["target","filepath"])
        clinical_test_dataframe = test_dataframe.copy().drop(columns=["target","filepath"])


        train_datagen_clinical = ClinicalDataGenerator(clinical_train_dataframe,clinical_train_target,batch_size=parameters['batch_size'])
        valid_datagen_clinical = ClinicalDataGenerator(clinical_valid_dataframe, clinical_valid_target,batch_size=parameters['batch_size'],shuffle=False)
        test_datagen_clinical = ClinicalDataGenerator(clinical_test_dataframe, clinical_test_target,batch_size=parameters['batch_size'],shuffle=False)
        n_classes = clinical_train_target.shape[1]
        clinical_input_num = clinical_train_dataframe.shape[1]

        # print("\n\n\n")
        #
        # print(clinical_train_dataframe)
        #
        # print("\n\n\n")
        steps_per_epoch = len(train_datagen_clinical)
        validation_steps = len(valid_datagen_clinical)
    else:
        parameters["clinical"]["depth"] = 0
        parameters["clinical"]["width_multiplier"] = 0
        parameters["clinical"]["activation_function"] = "relu"
        n_classes = 0
        n_classes = 0
        clinical_input_num = 0
        train_dataframe = train_dataframe.reset_index(drop=True)
        valid_dataframe = valid_dataframe.reset_index(drop=True)
        test_dataframe = test_dataframe.reset_index(drop=True)


############################################
    #choose preprocess function for the images
    if parameters['preprocessing_function']:
        if parameters['model_name'] == 'VGG16':
            preprocess_func = tf.keras.applications.vgg16.preprocess_input
            #target_size = (224,224)
        elif parameters['model_name'] == 'mobile_net':
            preprocess_func = tf.keras.applications.mobilenet_v2.preprocess_input
            #target_size = (224,224)
        elif parameters['model_name'] == 'xception':
            preprocess_func = tf.keras.applications.xception.preprocess_input
            #target_size = (299, 299)
        else:
            preprocess_func = None
            #target_size = (224,224)
    else:
        preprocess_func = None
    target_size = parameters['target_size']
    input_shape = (target_size[0],target_size[1],3)

############################################
    #prepare data generators for the images
    if parameters['image_model']:
        if parameters["data_augmentation"]:
            # Data augmentation
            train_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_func,
                rescale=1. / 255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(preprocessing_function=preprocess_func,
                                               rescale=1. / 255)

        validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_func,
                                                  rescale=1. / 255)
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_func,
                                                  rescale=1. / 255)
        if parameters['dataframe']:
            # dataframe = pd.read_csv(parameters['dataframe_path'],sep="\t")
            # train_dataframe, test_dataframe = train_test_split(dataframe, test_size=parameters['validation_split'], stratify=dataframe["target"])

            train_generator = train_datagen.flow_from_dataframe(
                train_dataframe,
                x_col=parameters['x_col'],
                y_col=parameters['y_col'],
                target_size=target_size,
                batch_size=parameters['batch_size'],
                class_mode='categorical'#,
                #save_to_dir = parameters['save_to_dir_train']#,
                #shuffle=True
                )

            validation_generator = validation_datagen.flow_from_dataframe(
                valid_dataframe,
                x_col=parameters['x_col'],
                y_col=parameters['y_col'],
                target_size=target_size,
                batch_size=parameters['batch_size'],
                class_mode='categorical',
                shuffle=False#,
                #save_to_dir=parameters['save_to_dir_validation']
                )

            test_generator = test_datagen.flow_from_dataframe(
                test_dataframe,
                x_col=parameters['x_col'],
                y_col=parameters['y_col'],
                target_size=target_size,
                batch_size=parameters['batch_size'],
                class_mode='categorical',
                shuffle=False#,
                #save_to_dir=parameters['save_to_dir_validation']
                )

        else:
            train_generator = train_datagen.flow_from_directory(parameters["trainDir"],
                                                                batch_size=parameters["batch_size"],
                                                                class_mode='categorical',
                                                                target_size=target_size,
                                                                save_to_dir = parameters['save_to_dir_train'])

            validation_generator = validation_datagen.flow_from_directory(parameters["valDir"],
                                                                          batch_size=parameters["batch_size"],
                                                                          class_mode='categorical',
                                                                          target_size=target_size,
                                                                          shuffle=False,
                                                                          save_to_dir=parameters['save_to_dir_validation'])

            test_generator = test_datagen.flow_from_directory(parameters["testDir"],
                                                                          batch_size=parameters["batch_size"],
                                                                          class_mode='categorical',
                                                                          target_size=target_size,
                                                                          shuffle=False,
                                                                          save_to_dir=parameters['save_to_dir_validation'])

        steps_per_epoch = math.ceil(train_generator.n / parameters["batch_size"])
        validation_steps = math.ceil(validation_generator.n / parameters["batch_size"])
        n_classes = len(train_generator.class_indices)
    print("\n\n\n")
    print((input_shape))
        # print(train_generator.filenames)
        # print(train_generator.class_indices)
        # print(train_generator.samples)
        # print(train_generator.n)
        # print(train_generator.__len__())
        # print(len(train_generator))
        # print("\n")
    #################################
    # for the use of multigpu
    #prepare the model
    #parameters["image_model"] = False
    #parameters["clinical_model"] = False
    if parameters["n_gpus"] > 1:
        device_type = 'GPU'
        devices = tf.config.experimental.list_physical_devices(
            device_type)
        devices_names = [d.name.split("e:")[1] for d in devices]

        strategy = tf.distribute.MirroredStrategy(
            devices=devices_names[:parameters["n_gpus"]])

        with strategy.scope():
            model = build_model(parameters["dropout"],parameters["learning_rate"],n_classes, parameters["fine_tune"], parameters["model_name"],input_shape,
                                parameters["clinical"]["depth"],parameters["clinical"]["width_multiplier"],
                                parameters["clinical"]["activation_function"],
                                parameters["image_model"],parameters["clinical_model"],clinical_input_num)
            #model = build_model(parameters["learning_rate"],n_classes, parameters["fine_tune"], parameters["model_name"],parameters["target_size"])
    else:
        model = build_model(parameters["dropout"],parameters["learning_rate"], n_classes, parameters["fine_tune"], parameters["model_name"],input_shape,
                            parameters["clinical"]["depth"], parameters["clinical"]["width_multiplier"],
                            parameters["clinical"]["activation_function"],
                            parameters["image_model"], parameters["clinical_model"], clinical_input_num)
        #model = build_model(parameters["learning_rate"],n_classes, parameters["fine_tune"], parameters["model_name"],parameters["target_size"])






    #################################
    #combined generator
    if parameters["image_model"] and parameters["clinical_model"]:
        train_generator_definitive = JoinedGen(train_generator, train_datagen_clinical)
        validation_generator_definitive = JoinedGen(validation_generator, valid_datagen_clinical)
        test_generator_definitive = JoinedGen(test_generator, test_datagen_clinical)
    elif parameters["clinical_model"] and not parameters["image_model"]:
        train_generator_definitive = train_datagen_clinical
        validation_generator_definitive = valid_datagen_clinical
        test_generator_definitive = test_datagen_clinical
    else:
        train_generator_definitive = train_generator
        validation_generator_definitive = validation_generator
        test_generator_definitive = test_generator
    #################################
    if parameters["class_weights"]:
            if parameters["balance_data"]:
                print("Not applying weights as data is already balanced")
                class_weight = None
            else:
                print("applying weights")
                class_weight = calculate_class_weights(train_generator_definitive)
    else:
        class_weight = None
    #################################
    #prepare callbacks
    if parameters["log"]:
        file_writer = tf.summary.create_file_writer(log_dir)
        callbacks = []
        if parameters["callbacks"]["tensorboard"]:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            callbacks.append(tensorboard_callback)
        if parameters["callbacks"]["checkpoint"]:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=log_dir + "/best-{epoch}",
                monitor='val_accuracy',
                save_best_only=True)
            callbacks.append(model_checkpoint_callback)

        if parameters["callbacks"]["early_stopping"]:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                              patience=parameters["callbacks_data"]["early_stopping_patience"])
            callbacks.append(early_stopping)

        if parameters["callbacks"]["reduce_lr"]:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                             factor=parameters["callbacks_data"]["reduce_lr_factor"],
                                                            patience=parameters["callbacks_data"]["reduce_lr_patience"],
                                                             min_lr=parameters["callbacks_data"]["reduce_lr_min_lr"])
            callbacks.append(reduce_lr)

        if parameters["callbacks"]["log"]:
            callbacks.append(log_learning_rate_callback(file_writer))
            #lr_log = keras.callbacks.LambdaCallback(on_epoch_end=log_learning_rate)
            #callbacks.append(lr_log)
        if parameters["callbacks"]["confusion_matrix"]:
            callbacks.append(confusion_matrix_callback(file_writer,validation_generator_definitive,train_generator_definitive))
            #cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix(self=model))

        #callbacks = [tensorboard_callback, cm_callback,reduce_lr,early_stopping,model_checkpoint_callback,lr_log]
    else:
        callbacks = []
    #################################

    if nni_activated:
        verbose = 1
        callbacks.append(log_nni_callback())
    else:
        verbose = 1

    print("validation steps: "+str(validation_steps))
    print("steps per epoch: "+str(steps_per_epoch))
    print("\n\n\n")

    model.fit(
        train_generator_definitive,
        validation_data=validation_generator_definitive,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=parameters["epochs"],
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=verbose#,
        #use_multiprocessing=True,
        #workers=parameters["workers"]
        #,drop_remainder=True
    )

    print(train_generator_definitive[1][0])
    print(model.predict(train_generator_definitive[1][0]))

    #test_acc = test(args, model, device, test_loader)
    callbacks2 = []
    if parameters["log_final"]:
        file_writer = tf.summary.create_file_writer(log_dir + "/test")
        callbacks2.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir + "/test", histogram_freq=1))
        callbacks2.append(confusion_matrix_test_callback(file_writer, test_generator_definitive))
    _,test_acc,_ = model.evaluate(test_generator_definitive,callbacks=callbacks2)
    print("Test accuracy:", test_acc)
    nni.report_final_result(test_acc)

    model.save(log_dir+"/final_model")

