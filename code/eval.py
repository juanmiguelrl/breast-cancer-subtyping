import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import itertools
import numpy as np
import sklearn.metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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

def evaluate_ann(model_dir,valDir,batch_size):
    model = tf.keras.models.load_model(model_dir + "model")
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


    val_ds = tf.keras.utils.image_dataset_from_directory(
      valDir,
      seed=123,
      image_size=(200, 200),
      batch_size=batch_size)


###############################################################################
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(valDir, batch_size=batch_size,
                                                                  class_mode='binary',
                                                                  target_size=(200, 200))

    true_labels = validation_generator.classes
    predictions = model.predict(validation_generator)

    y_true = true_labels
    y_pred = np.array([np.argmax(x) for x in predictions])

    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)

    print(cm)

    plot_confusion_matrix(cm, validation_generator.class_indices.keys())
    plt.show()



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