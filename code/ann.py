# %%

# Load libraries
from tensorflow.keras.applications import VGG16
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import datetime
import argparse
from tensorflow.python.client import device_lib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import itertools
import io
from tensorflow import keras
import sklearn.metrics
from tensorflow.keras.applications.vgg16 import decode_predictions


def train_ann( trainDir, valDir, logdir, batch_size, epochs, n_gpus):
  print(tf.config.list_physical_devices())

  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
  print(device_lib.list_local_devices())

  #for the use of multigpu
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
    modelFE = tf.keras.models.Sequential([pre_trained_model,
                                          tf.keras.layers.Flatten(),
                                          tf.keras.layers.Dense(256, activation = 'relu'),
                                          tf.keras.layers.Dense(1, activation = 'sigmoid')                                     
    ])
    modelFE.summary()

    # Compile the model
    modelFE.compile(loss='binary_crossentropy',
                    optimizer = tf.keras.optimizers.RMSprop(lr = 1e-4),
                    metrics=['acc'])

  # Data augmentation
  train_datagen = ImageDataGenerator(
      rescale = 1./255,
      rotation_range = 40,
      width_shift_range = 0.2,
      height_shift_range = 0.2,
      shear_range = 0.2,
      zoom_range = 0.2,
      horizontal_flip = True,
      fill_mode = 'nearest'
  )

  validation_datagen = ImageDataGenerator(rescale = 1./255)

  train_generator = train_datagen.flow_from_directory(trainDir,
                                                      batch_size = batch_size,
                                                      class_mode = 'binary',
                                                      target_size = (224, 224))
  validation_generator = validation_datagen.flow_from_directory(valDir, batch_size = batch_size,
                                                class_mode = 'binary',
                                                target_size = (224, 224))


    # #show train data images with its class
    # for data_batch, labels_batch in train_generator:
    #     print('data batch shape:', data_batch.shape)
    #     print('labels batch shape:', labels_batch.shape)
    #     break
    # #plots images from train data with labels
    # import matplotlib.pyplot as plt
    # for i in range(20):
    #     plt.plot()
    #     plt.imshow(data_batch[i])
    #     plt.title(labels_batch[i])
    #     plt.axis('off')
    #     plt.show()

    
  steps_per_epoch = train_generator.n // batch_size
  validation_steps = validation_generator.n // batch_size
  #print(steps_per_epoch)
  #print(validation_steps)

##########################################################


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
  #logging

  #log_dir = logdir + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
  log_dir = os.path.join(logdir,datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))
  #pass log_dir to path to normalize slashes
  #path_dir = Path((os.path.join(log_dir)))
  #log_dir = str(path_dir)
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


  # Creates a file writer for the log directory.
  file_writer = tf.summary.create_file_writer(log_dir)
  #file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')

  # Using the file writer, log the reshaped image.
  # with file_writer.as_default():
  #     # Don't forget to reshape.
  #     images = np.reshape(train_datagen[0:25], (-1, 28, 28, 1))
  #     tf.summary.image("25 training data examples", images, max_outputs=25, step=0)
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

  def log_confusion_matrix(epoch, logs):
      # Use the model to predict the values from the validation dataset.
      test_pred_raw = modelFE.predict(validation_generator)
      test_pred = np.argmax(test_pred_raw, axis=1)
      class_labels = list(validation_generator.class_indices.keys())

      ###########################
      results = decode_predictions(test_pred_raw)
      for result in results[0]:
          print(result[2])  # prints the accuracy levels of each class
      ###########################

      # Calculate the confusion matrix.
      cm = sklearn.metrics.confusion_matrix(validation_generator.classes, test_pred)
      # Log the confusion matrix as an image summary.
      figure = plot_confusion_matrix(cm, class_names=class_labels)
      cm_image = plot_to_image(figure)

      # Log the confusion matrix as an image summary.
      with file_writer.as_default():
          tf.summary.image("Confusion Matrix", cm_image, step=epoch)

  # Define the per-epoch callback.
  cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)




  ##############################################
  # ''' confusion matrix summaries '''
  # img_d_summary_dir = os.path.join(checkpoint_dir, "summaries", "img")
  # img_d_summary_writer = tf.summary.FileWriter(img_d_summary_dir, sess.graph)
  # img_d_summary = plot_confusion_matrix(correct_labels, predict_labels, labels, tensor_name='dev/cm')
  # img_d_summary_writer.add_summary(img_d_summary, current_step)

  ##############################################


  historyFE = modelFE.fit(
      train_generator,
      validation_data = validation_generator,
      steps_per_epoch = steps_per_epoch,
      epochs = epochs,
      validation_steps = validation_steps,
      callbacks=[tensorboard_callback,cm_callback]
  )


# import matplotlib.pyplot as plt
# # Plot results
# print(historyFE.history.keys())
# # summarize history for accuracy
# plt.plot(historyFE.history['acc'])
# plt.plot(historyFE.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
#
# # summarize history for loss
# plt.plot(historyFE.history['loss'])
# plt.plot(historyFE.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


