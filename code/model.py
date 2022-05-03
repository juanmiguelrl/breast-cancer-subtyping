import tensorflow as tf
from tensorflow.keras.applications import VGG16


def VGG16_model(learning_rate, n_classes):
    # Load VGG16 trained params and CNN network
    # pre_trained_model = VGG16(input_shape=(224, 224, 3),
    #                           include_top=False,
    #                           weights='imagenet')
    # pre_trained_model.trainable = True
    # set_trainable = False
    #
    # for layer in pre_trained_model.layers:
    #     if layer.name == 'block5_conv1':
    #         set_trainable = True
    #     if set_trainable:
    #         layer.trainable = True
    #     else:
    #         layer.trainable = False
    #
    # pre_trained_model.summary()

    # #############################
    # # Create the model
    # global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    # #prediction_layer = tf.keras.layers.Dense(1)
    # inputs = tf.keras.Input(shape=(224, 224, 3))
    # x = tf.keras.applications.vgg16.preprocess_input(inputs, data_format=None)
    # x = pre_trained_model(x)
    # x = global_average_layer(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.Flatten() (x)
    # x = tf.keras.layers.Dense(256, activation='relu') (x)
    # x = tf.keras.layers.Dense(n_classes, activation='softmax') (x)
    # #outputs = prediction_layer(x)
    # #outputs = tf.keras.layers.Dense(n_classes, activation='softmax') (outputs)
    # model = tf.keras.Model(inputs, x)
    #############################
    # inputs = tf.keras.Input(shape=(224, 224, 3))
    # x = tf.keras.applications.vgg16.preprocess_input(inputs, data_format=None)
    # flatten_layer = tf.keras.layers.Flatten()
    # dense_layer_1 = tf.keras.layers.Dense(50, activation='relu')
    # dense_layer_2 = tf.keras.layers.Dense(20, activation='relu')
    # prediction_layer = tf.keras.layers.Dense(5, activation='softmax')
    # model = tf.keras.models.Sequential([
    #     pre_trained_model(x),
    #     flatten_layer,
    #     dense_layer_1,
    #     dense_layer_2,
    #     prediction_layer
    # ])
    # 2 full conected layers to be trained are added to the model
    # model = tf.keras.models.Sequential([#tf.keras.Input(shape=(224, 224, 3)),
    #                                     pre_trained_model,
    #                                     tf.keras.layers.Flatten(),
    #                                     tf.keras.layers.Dense(256, activation='relu'),
    #                                     tf.keras.layers.Dense(n_classes, activation='softmax')
    #                                     ])
    # model.summary()
    #
    # # Compile the model
    # model.compile(loss='sparse_categorical_crossentropy',
    #               optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
    #               metrics=['acc'])
    # return model

    #############################
    fine_tune = 2
    conv_base = VGG16(input_shape=(224, 224, 3),
                              include_top=False,
                              weights='imagenet')

    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    # Create a new 'top' of the model (i.e. fully-connected layers).
    # This is 'bootstrapping' a new top_model onto the pretrained layers.
    top_model = conv_base.output
    top_model = tf.keras.layers.Flatten(name="flatten")(top_model)
    top_model = tf.keras.layers.Dense(4096, activation='relu')(top_model)
    top_model = tf.keras.layers.Dense(1072, activation='relu')(top_model)
    top_model = tf.keras.layers.Dropout(0.2)(top_model)
    output_layer = tf.keras.layers.Dense(n_classes, activation='softmax')(top_model)

    # Group the convolutional base and new fully-connected layers into a Model object.
    model = tf.keras.models.Model(inputs=conv_base.input, outputs=output_layer)

    # Compiles the model for training.
    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
