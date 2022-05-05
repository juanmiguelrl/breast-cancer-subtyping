import tensorflow as tf
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import itertools
import numpy as np
import sklearn.metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
import io

class confusion_matrix_callback(keras.callbacks.Callback):
    def __init__(self,file_writer,validation_generator,train_generator):
        self.file_writer = file_writer
        self.validation_generator = validation_generator
        self.train_generator = train_generator
    def on_epoch_end(self, epoch, logs=None):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = self.model.predict(self.validation_generator)
        test_pred = np.argmax(test_pred_raw, axis=1)
        #class_labels = list(val_ds.class_indices.keys())

        # Calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix(self.validation_generator.classes, test_pred)
        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(cm, self.validation_generator.class_indices.keys())
        cm_image = plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with self.file_writer.as_default():
            tf.summary.image("Validation Confusion Matrix", cm_image, step=epoch)

        ###################        #repeat the same but with the training data
        # Use the model to predict the values from the validation dataset.
        test_pred_rawt = self.model.predict(self.train_generator)
        test_predt = np.argmax(test_pred_rawt, axis=1)
        #class_labels = list(val_ds.class_indices.keys())

        # Calculate the confusion matrix.
        cmt = sklearn.metrics.confusion_matrix(self.train_generator.classes, test_predt)
        # Log the confusion matrix as an image summary.
        figuret = plot_confusion_matrix(cmt, self.train_generator.class_indices.keys())
        cm_imaget = plot_to_image(figuret)

        # Log the confusion matrix as an image summary.
        with self.file_writer.as_default():
            tf.summary.image("Training Confusion Matrix", cm_imaget, step=epoch)

class log_learning_rate_callback(keras.callbacks.Callback):
    def __init__(self,file_writer):
        self.file_writer = file_writer
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        with self.file_writer.as_default():
            tf.summary.scalar('learning rate', lr, step=epoch)

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def evaluate_ann(model_dir,valDir,batch_size,log_dir=None):
    model = tf.keras.models.load_model(model_dir)


    # val_ds = tf.keras.utils.image_dataset_from_directory(
    #   valDir,
    #   seed=123,
    #   image_size=(200, 200),
    #   batch_size=batch_size)


###############################################################################
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(valDir, batch_size=batch_size,
                                                                  class_mode='binary',
                                                                  target_size=(224, 224))

    true_labels = validation_generator.classes
    predictions = model.predict(validation_generator)

    y_true = true_labels
    y_pred = np.array([np.argmax(x) for x in predictions])

    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)

    print(cm)

    figure = plot_confusion_matrix(cm, validation_generator.class_indices.keys())
    #plt.show()

    # Log the confusion matrix as an image summary.
    cm_image = plot_to_image(figure)

    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(log_dir)
    # Log the confusion matrix as an image summary.
    with file_writer.as_default():
        tf.summary.image("Confusion Matrix", cm_image,step=0)



    # cm = tf.math.confusion_matrix([0,1,2,3] , np.argmax(model.predict(val_ds)))
    # print(cm)

    # test_pred_raw = model.predict(val_ds)
    # test_pred = np.argmax(test_pred_raw, axis=1)
    # print(test_pred)
    # test_labels = np.array(val_ds.class_names)
    # print(test_labels)
    # print(test_pred.shape)
    # print(test_labels.shape)
    #
    # # Compute confusion matrix
    # cnf_matrix = sklearn.metrics.confusion(test_labels, test_pred)
    # cm = sklearn.metrics.confusion_matrix(val_ds.class_names, test_pred)
    # figure = plot_confusion_matrix(cm, class_names=val_ds.class_names)
    # plt.show(figure)


#tensorboard --logdir logs / train_data