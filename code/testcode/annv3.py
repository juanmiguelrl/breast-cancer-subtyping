import numpy as np
from tensorflow import keras

import sklearn.metrics
from eval import  plot_confusion_matrix,plot_to_image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import calculate_class_weights

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import conv_model


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.use('Agg')


#trainDir = "/mnt/netapp2/Store_uni/home/ulc/co/jrl/ai/dog_cat_data/training_set"
#valDir = "/mnt/netapp2/Store_uni/home/ulc/co/jrl/ai/dog_cat_data/test_set"
trainDir = "/mnt/netapp2/Store_uni/home/ulc/co/jrl/imgdir/images_together_filtered_classified/train"
valDir = "/mnt/netapp2/Store_uni/home/ulc/co/jrl/imgdir/images_together_filtered_classified/test"
batch_size = 32

def train_ann( trainDir, valDir,batch_size):


    train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input,rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True
                                           )

    training_set = train_datagen.flow_from_directory(trainDir,batch_size=batch_size,
                                                         target_size=(224, 224))

    test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
                                          rescale=1. / 255)

    test_set = test_datagen.flow_from_directory(valDir,
                                                    target_size=(224, 224),
                                                    batch_size=batch_size,
                                                    shuffle=False)

    n_classes = training_set.num_classes
    #model = conv_model(learning_rate, n_classes, 50)

    vgg16_model = tf.keras.applications.vgg16.VGG16()

    model = Sequential()
    for layer in vgg16_model.layers[:-1]:
        model.add(layer)
    for layer in model.layers:
        layer.trainable = False
    model.add(Dense(units=n_classes, activation='softmax'))

    # model = Sequential([
    #         Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    #         MaxPool2D(pool_size=(2, 2), strides=2),
    #         Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    #         MaxPool2D(pool_size=(2, 2), strides=2),
    #         Flatten(),
    #         Dense(units=n_classes, activation='softmax')
    # ])

    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    class_weight = calculate_class_weights(training_set)
    model.fit(x=training_set,
              steps_per_epoch=len(training_set),
              epochs=10,
              class_weight=class_weight
              )

    predictions = model.predict(x = test_set, steps = len(test_set), verbose = 0)

    np.round(predictions)

    cm = confusion_matrix(y_true=test_set.classes, y_pred=np.argmax(predictions, axis=-1))

    print(cm)

train_ann(trainDir, valDir, batch_size)
    # train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    #     .flow_from_directory(directory=trainDir, target_size=(224, 224), batch_size=10)
    # test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    #     .flow_from_directory(directory=valDir, target_size=(224, 224), batch_size=10, shuffle=False)
    #
    # imgs, labels = next(train_batches)

    # def plotImages(images_arr):
    #    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    #    axes = axes.flatten()
    #    for img, ax in zip( images_arr, axes):
    #        ax.imshow(img)
    #        ax.axis('off')
    #    plt.tight_layout()
    #    plt.show()

    # plotImages(imgs)
    # print(labels)

    # model = Sequential([
    #     Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    #     MaxPool2D(pool_size=(2, 2), strides=2),
    #     Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    #     MaxPool2D(pool_size=(2, 2), strides=2),
    #     Flatten(),
    #     Dense(units=4, activation='softmax')
    # ])

    # model.summary()
    #
    # model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # model.fit(x=train_batches, epochs=5)

    # test_imgs, test_labels = next(test_batches)
    # # plotImages(test_imgs)
    # print(test_labels)
    #
    # predictions = model.predict(x=test_batches, verbose=0)
    #
    # np.round(predictions)
    #
    # cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
    #
    # print(cm)


#     steps_per_epoch = training_set.n // batch_size
#     validation_steps = test_set.n // batch_size
#     ########################################
#     #logging
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#     file_writer = tf.summary.create_file_writer(log_dir)
#     # Define the per-epoch callback.
#
#     def log_confusion_matrix(epoch, logs):
#         # Use the model to predict the values from the validation dataset.
#         test_pred_raw = model.predict(test_set)
#         test_pred = np.argmax(test_pred_raw, axis=1)
#         #class_labels = list(val_ds.class_indices.keys())
#
#         # Calculate the confusion matrix.
#         cm = sklearn.metrics.confusion_matrix(test_set.classes, test_pred)
#         # Log the confusion matrix as an image summary.
#         figure = plot_confusion_matrix(cm, test_set.class_indices.keys())
#         cm_image = plot_to_image(figure)
#
#         # Log the confusion matrix as an image summary.
#         with file_writer.as_default():
#             tf.summary.image("Validation Confusion Matrix", cm_image, step=epoch)
#
#         ###################
#         #repeat the same but with the training data
#         # Use the model to predict the values from the validation dataset.
#         test_pred_rawt = model.predict(training_set)
#         test_predt = np.argmax(test_pred_rawt, axis=1)
#         #class_labels = list(val_ds.class_indices.keys())
#
#         # Calculate the confusion matrix.
#         cmt = sklearn.metrics.confusion_matrix(training_set.classes, test_predt)
#         # Log the confusion matrix as an image summary.
#         figuret = plot_confusion_matrix(cmt, training_set.class_indices.keys())
#         cm_imaget = plot_to_image(figuret)
#
#         # Log the confusion matrix as an image summary.
#         with file_writer.as_default():
#             tf.summary.image("Training Confusion Matrix", cm_imaget, step=epoch)
#
#
#
#     cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
# ########################################
#     def log_learning_rate(epoch, logs):
#         lr = model.optimizer.learning_rate
#         with file_writer.as_default():
#             tf.summary.scalar('learning rate', lr, step=epoch)
#
#     lr_log = keras.callbacks.LambdaCallback(on_epoch_end=log_learning_rate)
# ########################################
#     reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
#                                   patience=5, min_lr=0)
#     early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=15)
#
#     model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#         filepath=logdir + "/best-{epoch}",
#         monitor='val_accuracy',
#         save_best_only=True)
# ########################################
#     steps_per_epoch = training_set.n // batch_size
#     validation_steps = test_set.n // batch_size
#
#
    # class_weight = calculate_class_weights(training_set)
    # if log:
    #     callbacks = [tensorboard_callback, cm_callback,reduce_lr,early_stopping,model_checkpoint_callback,lr_log]
    # else:
    #     callbacks = None
    #
    # result = model.fit(training_set, validation_data = test_set, epochs = 6,
    #                    callbacks=callbacks,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps
    #                    )
    #
    # model.save(model_dir)

    # print("asodfawjhdfahdoìhadihj")
    # result = model.fit(training_set, validation_data = test_set, epochs = 6)
    # model.save(model_dir)

