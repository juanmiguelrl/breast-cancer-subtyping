import pandas as pd

from model import build_model
from clinical_model import process_clinical_data,ClinicalDataGenerator,JoinedGen
import numpy as np
import tensorflow as tf
import nni
import math
from tensorflow import keras

import sklearn.metrics
from eval import  plot_confusion_matrix,plot_to_image,confusion_matrix_callback,log_learning_rate_callback, log_nni_callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import calculate_class_weights
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split

def train_ann( parameters,model_dir,log_dir,nni_activated):

    print(tf.config.list_physical_devices())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(device_lib.list_local_devices())

############################################
    #dataframe
    dataframe = pd.read_csv(parameters['dataframe_path'],sep="\t")
    train_dataframe, test_dataframe = train_test_split(dataframe, test_size=parameters['validation_split'], stratify=dataframe["target"])



############################################
    #prepare data generators with clinical data
    if parameters["clinical_model"]:
        clinical_train_dataframe,clinical_train_target = process_clinical_data(train_dataframe,parameters["clinical_columns"])
        clinical_test_dataframe, clinical_test_target = process_clinical_data(test_dataframe,parameters["clinical_columns"])
        train_datagen_clinical = ClinicalDataGenerator(clinical_train_dataframe,clinical_train_target,batch_size=parameters['batch_size'])
        test_datagen_clinical = ClinicalDataGenerator(clinical_test_dataframe, clinical_test_target,batch_size=parameters['batch_size'],shuffle=False)
        #train_clinical,test_clinical = load_clinical_data(parameters["clinical_columns"],train_dataframe,test_dataframe,dataframe)
        n_classes = clinical_train_target.shape[1]
        clinical_input_num = clinical_train_dataframe.shape[1]

        # print("\n\n\n")
        #
        # print(clinical_train_dataframe)
        #
        # print("\n\n\n")
    else:
        n_classes = 0
        clinical_input_num = 0



############################################
    #choose preprocess function for the images
    if parameters['preprocessing_function']:
        if parameters['model_name'] == 'VGG16':
            preprocess_func = tf.keras.applications.vgg16.preprocess_input
            target_size = (224,224)
        elif parameters['model_name'] == 'mobile_net':
            preprocess_func = tf.keras.applications.mobilenet_v2.preprocess_input
            target_size = (224,224)
        elif parameters['model_name'] == 'xception':
            preprocess_func = tf.keras.applications.xception.preprocess_input
            target_size = (299, 299)
        else:
            preprocess_func = None
            target_size = (224,224)
    else:
        preprocess_func = None
        target_size = parameters['target_size']


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
        if parameters['dataframe']:
            dataframe = pd.read_csv(parameters['dataframe_path'],sep="\t")
            train_dataframe, test_dataframe = train_test_split(dataframe, test_size=parameters['validation_split'], stratify=dataframe["target"])

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

            validation_generator = validation_datagen.flow_from_directory(parameters["testDir"],
                                                                          batch_size=parameters["batch_size"],
                                                                          class_mode='categorical',
                                                                          target_size=target_size,
                                                                          shuffle=False,
                                                                          save_to_dir=parameters['save_to_dir_validation'])

        n_classes = len(train_generator.class_indices)

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
            model = build_model(parameters["learning_rate"],n_classes, parameters["fine_tune"], parameters["model_name"],(target_size,3),
                                parameters["image_model"],parameters["clinical_model"],clinical_input_num)
            #model = build_model(parameters["learning_rate"],n_classes, parameters["fine_tune"], parameters["model_name"],parameters["target_size"])
    else:
        model = build_model(parameters["learning_rate"], n_classes, parameters["fine_tune"], parameters["model_name"],(target_size,3),
                            parameters["image_model"], parameters["clinical_model"], clinical_input_num)
        #model = build_model(parameters["learning_rate"],n_classes, parameters["fine_tune"], parameters["model_name"],parameters["target_size"])

    steps_per_epoch = math.ceil(train_generator.n / parameters["batch_size"])
    validation_steps = math.ceil(validation_generator.n / parameters["batch_size"])


    if parameters["class_weights"]:
        class_weight = calculate_class_weights(train_generator)
    else:
        class_weight = None

    #################################
    #combined generator
    if parameters["image_model"] and parameters["clinical_model"]:
        train_generator_definitive = JoinedGen(train_generator, train_datagen_clinical)
        validation_generator_definitive = JoinedGen(validation_generator, test_datagen_clinical)
    elif parameters["clinical_model"] and not parameters["image_model"]:
        train_generator_definitive = train_datagen_clinical
        validation_generator_definitive = test_datagen_clinical
    else:
        train_generator_definitive = train_generator
        validation_generator_definitive = validation_generator
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
        verbose = 0
        callbacks.append(log_nni_callback())
    else:
        verbose = 1

    model.fit(
        train_generator_definitive,
        validation_data=validation_generator_definitive,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=parameters["epochs"],
        callbacks=callbacks,
        #class_weight=class_weight,
        verbose=verbose
    )

    print(model.predict(train_generator_definitive[1][0]))

    #test_acc = test(args, model, device, test_loader)
    _,test_acc = model.evaluate(validation_generator_definitive)
    print("Test accuracy:", test_acc)
    nni.report_final_result(test_acc)

    model.save(model_dir)

