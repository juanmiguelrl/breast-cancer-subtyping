import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.client import device_lib

from datetime import datetime
import io
import itertools
from packaging import version

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics



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

    (train_images, train_labels), (test_images, test_labels) = \
        (train_ds,train_ds.class_names), (val_ds,val_ds.class_names)


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

    num_classes = 4

    model = tf.keras.Sequential([
      tf.keras.layers.Rescaling(1./255),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3
    )

    model.save(model_dir + "model")

