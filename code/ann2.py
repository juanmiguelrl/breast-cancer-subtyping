import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.client import device_lib
from model import VGG16_model

from datetime import datetime
import io
import itertools
from packaging import version

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from eval import  plot_confusion_matrix,plot_to_image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_ann( trainDir, valDir, logdir, batch_size, epochs, n_gpus,model_dir,
               learning_rate,n_classes,log_dir=None):
    from tensorflow.python.client import device_lib
    print(tf.config.list_physical_devices())

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(device_lib.list_local_devices())

#################################

    # train_ds = tf.keras.utils.image_dataset_from_directory(
    #     trainDir,
    #     seed=123,
    #     image_size=(200, 200),
    #     batch_size=batch_size)
    #
    # val_ds = tf.keras.utils.image_dataset_from_directory(
    #   valDir,
    #   seed=123,
    #   image_size=(200, 200),
    #   batch_size=batch_size)
    #
    # class_names = train_ds.class_names
    # print(class_names)


#################################
    # for the use of multigpu
    if n_gpus > 1:
        device_type = 'GPU'
        devices = tf.config.experimental.list_physical_devices(
            device_type)
        devices_names = [d.name.split("e:")[1] for d in devices]

        strategy = tf.distribute.MirroredStrategy(
            devices=devices_names[:n_gpus])

        with strategy.scope():
            model = VGG16_model(learning_rate,n_classes)
    else:
        model = VGG16_model(learning_rate,n_classes)

########################################
    #logging
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    file_writer = tf.summary.create_file_writer(log_dir)
    # Define the per-epoch callback.


    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_generator = validation_datagen.flow_from_directory(trainDir,
                                                        batch_size=batch_size,
                                                        class_mode='binary',
                                                        target_size=(224, 224))
    validation_generator = validation_datagen.flow_from_directory(valDir, batch_size=batch_size,
                                                                  class_mode='binary',
                                                                  target_size=(224, 224))
    ###################
    ###################
    def log_confusion_matrix(epoch, logs):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = model.predict(validation_generator)
        test_pred = np.argmax(test_pred_raw, axis=1)
        #class_labels = list(val_ds.class_indices.keys())

        # Calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix(validation_generator.classes, test_pred)
        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(cm, validation_generator.class_indices.keys())
        cm_image = plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with file_writer.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)




    cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

########################################

    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size

    model.fit(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[tensorboard_callback, cm_callback]
    )

    model.save(model_dir)

