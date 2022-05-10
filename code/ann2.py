from model import VGG16_model,mobile_net_model,build_model,conv_model2
import numpy as np
import tensorflow as tf
import nni
from tensorflow import keras

import sklearn.metrics
from eval import  plot_confusion_matrix,plot_to_image,confusion_matrix_callback,log_learning_rate_callback, log_nni_callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import calculate_class_weights
from tensorflow.python.client import device_lib

def train_ann( parameters,model_dir,log_dir,nni_activated):

    print(tf.config.list_physical_devices())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(device_lib.list_local_devices())


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
        #target_size = (224,224)

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

    train_generator = train_datagen.flow_from_directory(parameters["trainDir"],
                                                        batch_size=parameters["batch_size"],
                                                        class_mode='categorical',
                                                        target_size=parameters["target_size"])

    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_func,
                                          rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(parameters["testDir"],
                                                                  batch_size=parameters["batch_size"],
                                                                  class_mode='categorical',
                                                                  target_size=parameters["target_size"],
                                                                  shuffle=False)

    n_classes = train_generator.num_classes
    #################################
    # for the use of multigpu
    if parameters["n_gpus"] > 1:
        device_type = 'GPU'
        devices = tf.config.experimental.list_physical_devices(
            device_type)
        devices_names = [d.name.split("e:")[1] for d in devices]

        strategy = tf.distribute.MirroredStrategy(
            devices=devices_names[:parameters["n_gpus"]])

        with strategy.scope():
            model = build_model(parameters["learning_rate"],n_classes, parameters["fine_tune"], parameters["model_name"],parameters["target_size"])
    else:
        model = build_model(parameters["learning_rate"],n_classes, parameters["fine_tune"], parameters["model_name"],parameters["target_size"])

    steps_per_epoch = train_generator.n // parameters["batch_size"]
    validation_steps = validation_generator.n // parameters["batch_size"]


    if parameters["class_weights"]:
        class_weight = calculate_class_weights(train_generator)
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
            callbacks.append(confusion_matrix_callback(file_writer,validation_generator,train_generator))
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
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=parameters["epochs"],
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=verbose
    )

    #test_acc = test(args, model, device, test_loader)
    test_acc = model.evaluate(validation_generator)
    print("Test accuracy:", test_acc)
    nni.report_final_result(test_acc)

    model.save(model_dir)

