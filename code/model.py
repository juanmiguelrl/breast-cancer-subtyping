import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, Adagrad

def VGG16_model(learning_rate, n_classes,fine_tune=0):
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


def VGG16_model2(learning_rate, n_classes,fine_tune=0):
    vgg16_model = tf.keras.applications.vgg16.VGG16()

    model = Sequential()
    for layer in vgg16_model.layers[:-1]:
        model.add(layer)
    for layer in model.layers:
        layer.trainable = False

    # if fine_tune > 0:
    #     for layer in model.layers[:-fine_tune]:
    #         layer.trainable = False
    # else:
    #     for layer in model.layers:
    #         layer.trainable = False

    model.add(Dense(units=n_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def conv_model2(learning_rate, n_classes,fine_tune=0):

    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    model = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
            MaxPool2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPool2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dense(units=n_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def mobile_net_model(learning_rate, n_classes,fine_tune=0):
    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
    mobile_net.trainable = False  # keeeping mobile net weights same

    trf_learn = Sequential()

    trf_learn.add(mobile_net)
    trf_learn.add(Flatten())

    trf_learn.add(Dense(units=32, activation="relu"))
    trf_learn.add(Dense(units=1, activation="sigmoid"))

    trf_learn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    trf_learn.summary()

    return trf_learn

def conv_model(learning_rate, n_classes,fine_tune=0):
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu", padding="valid", input_shape=[224, 224, 3]))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Flatten())

    model.add(Dense(units=32, activation="relu"))
    model.add(Dropout(0.4))

    model.add(Dense(units=1, activation="softmax"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

def build_model(learning_rate, n_classes,fine_tune=0,model_name="vgg16"):
    if model_name == "vgg16":
        return VGG16_model2(learning_rate, n_classes,fine_tune)
    elif model_name == "mobile_net":
        return mobile_net_model(learning_rate, n_classes,fine_tune)
    elif model_name == "conv":
        return conv_model(learning_rate, n_classes,fine_tune)
    else:
        return VGG16_model2(learning_rate, n_classes,fine_tune)