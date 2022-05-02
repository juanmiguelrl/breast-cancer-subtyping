import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.client import device_lib

from datetime import datetime
import io
import itertools
from packaging import version

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from eval import  plot_confusion_matrix,plot_to_image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_ann( trainDir, valDir, logdir, batch_size, epochs, n_gpus,model_dir,log_dir=None):
    from tensorflow.python.client import device_lib
    print(tf.config.list_physical_devices())

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(device_lib.list_local_devices())

    print("TensorFlow version: ", tf.__version__)
    assert version.parse(tf.__version__).release[0] >= 2, \
        "This notebook requires TensorFlow 2.0 or above."
    from tensorflow.python.client import device_lib

    # Download the data. The data is already divided into train and test.
    # The labels are integers representing classes.
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = \
        fashion_mnist.load_data()

    # # Names of the integer classes, i.e., 0 -> T-short/top, 1 -> Trouser, etc.
    # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #     'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



#################################

    train_ds = tf.keras.utils.image_dataset_from_directory(
        trainDir,
        seed=123,
        image_size=(200, 200),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
      valDir,
      seed=123,
      image_size=(200, 200),
      batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    # (train_images, train_labels), (test_images, test_labels) = \
    #     (train_ds,train_ds.class_names), (val_ds,val_ds.class_names)


    # Sets up a timestamped log directory.
    if log_dir is None:
        log_dir = logdir + "imgs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(log_dir)

    #
    # #get images from dataset and store them in an array
    # for images, labels in train_ds.take(1):
    #     images_array = images.numpy()
    #
    # # Using the file writer, log the reshaped image.
    # with file_writer.as_default():
    #     # Don't forget to reshape.
    #     images = np.reshape(images_array[0:25], (-1, 28, 28, 1))
    #     tf.summary.image("25 training data examples", images, max_outputs=25, step=0)

    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")
    #
    # plt.show()
    #
    # for image_batch, labels_batch in train_ds:
    #   print(image_batch.shape)
    #   print(labels_batch.shape)
    #   break


    normalization_layer = tf.keras.layers.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    # plt.figure(figsize=(10, 10))
    # for images, labels in normalized_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         #print the images but now normalized to [0,1]
    #         plt.imshow(images[i].numpy())
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")
    #
    # plt.show()



#################################
    # for the use of multigpu
    device_type = 'GPU'
    devices = tf.config.experimental.list_physical_devices(
        device_type)
    devices_names = [d.name.split("e:")[1] for d in devices]

    strategy = tf.distribute.MirroredStrategy(
        devices=devices_names[:n_gpus])

    with strategy.scope():

        # Load VGG16 trained params and CNN network
        pre_trained_model = VGG16(input_shape= (224, 224, 3),
                                  include_top = False,
                                  weights = 'imagenet')
        pre_trained_model.trainable = True
        set_trainable = False

        for layer in pre_trained_model.layers:
          if layer.name == 'block5_conv1':
            set_trainable = True
          if set_trainable:
            layer.trainable = True
          else:
            layer.trainable = False

        pre_trained_model.summary()

        #2 full conected layers to be trained are added to the model
        model = tf.keras.models.Sequential([pre_trained_model,
                                              tf.keras.layers.Flatten(),
                                              tf.keras.layers.Dense(256, activation = 'relu'),
                                              tf.keras.layers.Dense(4, activation = 'softmax')
        ])
        model.summary()

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy',
                        optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
                        metrics=['acc'])

        # model = keras.Sequential([
        #     keras.layers.Flatten(input_shape=(224, 224, 3)),
        #     keras.layers.Dense(128, activation='relu'),
        #     keras.layers.Dense(4, activation='softmax')
        # ])
        # model.compile(optimizer='adam',
        #               loss='sparse_categorical_crossentropy',
        #               metrics=['accuracy'])
    #################################
        # num_classes = 4
        #
        # model = tf.keras.Sequential([
        #   tf.keras.layers.Rescaling(1./255),
        #   tf.keras.layers.Conv2D(32, 3, activation='relu'),
        #   tf.keras.layers.MaxPooling2D(),
        #   tf.keras.layers.Conv2D(32, 3, activation='relu'),
        #   tf.keras.layers.MaxPooling2D(),
        #   tf.keras.layers.Conv2D(32, 3, activation='relu'),
        #   tf.keras.layers.MaxPooling2D(),
        #   tf.keras.layers.Flatten(),
        #   tf.keras.layers.Dense(128, activation='relu'),
        #   tf.keras.layers.Dense(num_classes)
        # ])

        # model.compile(
        #     optimizer='adam',
        #     loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        #     metrics=['accuracy'])

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

    model.save(model_dir + "model")

