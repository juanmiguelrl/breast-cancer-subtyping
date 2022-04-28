# %%

# Load libraries
from tensorflow.keras.applications import VGG16
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import datetime
import argparse
from tensorflow.python.client import device_lib


def train_ann(inputDir, trainDir, valDir, logdir, batch_size, epochs, n_gpus):
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


  #log_dir = logdir + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
  #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


  historyFE = modelFE.fit(
      train_generator,
      validation_data = validation_generator,
      steps_per_epoch = steps_per_epoch,
      epochs = epochs,
      validation_steps = validation_steps#,
      #callbacks=[tensorboard_callback]
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


