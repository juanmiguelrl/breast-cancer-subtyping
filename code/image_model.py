import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras import Input
from tensorflow.keras import Input,Model

def VGG16_model(dropout,learning_rate, n_classes,fine_tune,input_shape):
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
    if dropout > -1:
        top_model = tf.keras.layers.Dropout(dropout)(top_model)
    output_layer = tf.keras.layers.Dense(n_classes, activation='softmax')(top_model)

    # Group the convolutional base and new fully-connected layers into a Model object.
    model = tf.keras.models.Model(inputs=conv_base.input, outputs=output_layer)

    # Compiles the model for training.
    # model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate),
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])

    return model


def VGG16_model2(dropout,earning_rate, n_classes,fine_tune,input_shape):
    vgg16_model = tf.keras.applications.vgg16.VGG16(weights="imagenet", include_top=False)
    if fine_tune > 0:
        for layer in vgg16_model.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in vgg16_model.layers:
            layer.trainable = False
    # model = Sequential()
    # for layer in vgg16_model.layers[:-1]:
    #     model.add(layer)
    # for layer in model.layers:
    #     layer.trainable = False



    # if fine_tune > 0:
    #     for layer in model.layers[:-fine_tune]:
    #         layer.trainable = False
    # else:
    #     for layer in model.layers:
    #         layer.trainable = False

    # model.add(Dense(units=n_classes, activation='softmax'))
    # model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()

    # input = Input(shape=(224, 224, 3))
    # model = vgg16_model(input)
    # model = Dense(units=n_classes, activation='softmax') (model)
    # model = Model(inputs=input, outputs=model)

    #vgg16_model.trainable = False
    input = Input(shape=input_shape)
    model = vgg16_model(input)
    if dropout > -1:
        model = tf.keras.layers.Dropout(dropout)(model)
    model = tf.keras.layers.GlobalAveragePooling2D()(model)
    model = Dense(units=n_classes, activation='softmax') (model)
    model = Model(inputs=input, outputs=model)

    return model

# def conv_model2(dropout,learning_rate, n_classes,fine_tune=0):
#
#     if fine_tune > 0:
#         for layer in conv_base.layers[:-fine_tune]:
#             layer.trainable = False
#     else:
#         for layer in conv_base.layers:
#             layer.trainable = False
#
#     model = Sequential([
#             Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
#             MaxPool2D(pool_size=(2, 2), strides=2),
#             Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
#             MaxPool2D(pool_size=(2, 2), strides=2),
#             Flatten(),
#             Dense(units=n_classes, activation='softmax')
#     ])
#     # model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
#     # model.summary()
#     return model

def mobile_net_model(dropout,learning_rate, n_classes,fine_tune,input_shape):
    mobile_net = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False)
    #mobile_net.trainable = False  # keeeping mobile net weights same
    if fine_tune > 0:
        for layer in mobile_net.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in mobile_net.layers:
            layer.trainable = False
    # trf_learn = Sequential()
    #
    # trf_learn.add(mobile_net)
    # trf_learn.add(Flatten())
    #
    # trf_learn.add(Dense(units=32, activation="relu"))
    # trf_learn.add(Dense(units=1, activation="sigmoid"))

    # trf_learn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    #
    # trf_learn.summary()

    #mobile_net.trainable = False
    input = Input(shape=input_shape)
    model = mobile_net(input)
    if dropout > -1:
        model = tf.keras.layers.Dropout(dropout)(model)
    model = tf.keras.layers.GlobalAveragePooling2D()(model)
    model = Dense(units=n_classes, activation='softmax') (model)
    model = Model(inputs=input, outputs=model)

    return model

# def conv_model(dropout,learning_rate, n_classes,fine_tune=0):
#     model = Sequential()
#
#     model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu", padding="valid", input_shape=[224, 224, 3]))
#     model.add(MaxPool2D(pool_size=2, strides=2))
#
#     model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
#     model.add(MaxPool2D(pool_size=2, strides=2))
#
#     model.add(Flatten())
#
#     model.add(Dense(units=32, activation="relu"))
#     model.add(Dropout(0.4))
#
#     model.add(Dense(units=1, activation="softmax"))
#
#     # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#     # model.summary()
#     return model

#
# class Patches(tf.keras.layers.Layer):
#     def __init__(self, patch_size):
#         super(Patches, self).__init__()
#         self.patch_size = patch_size
#
#     def call(self, images):
#         batch_size = tf.shape(images)[0]
#         patches = tf.image.extract_patches(
#             images=images,
#             sizes=[1, self.patch_size, self.patch_size, 1],
#             strides=[1, self.patch_size, self.patch_size, 1],
#             rates=[1, 1, 1, 1],
#             padding="VALID",
#         )
#         patch_dims = patches.shape[-1]
#         patches = tf.reshape(patches, [batch_size, -1, patch_dims])
#         return patches
#
# class CreatePatches( tf.keras.layers.Layer ):
#
#   def __init__( self , patch_size ):
#     super( CreatePatches , self ).__init__()
#     self.patch_size = patch_size
#
#   def call(self, inputs ):
#     patches = []
#     # For square images only ( as inputs.shape[ 1 ] = inputs.shape[ 2 ] )
#     input_image_size = inputs.shape[ 1 ]
#     for i in range( 0 , input_image_size , self.patch_size ):
#         for j in range( 0 , input_image_size , self.patch_size ):
#             patches.append( inputs[ : , i : i + self.patch_size , j : j + self.patch_size , : ] )
#     return patches
#
# class PatchEncoder(tf.keras.layers.Layer):
#     def __init__(self, num_patches, projection_dim):
#         super(PatchEncoder, self).__init__()
#         self.num_patches = num_patches
#         self.projection = tf.keras.layers.Dense(units=projection_dim)
#         self.position_embedding = tf.keras.layers.Embedding(
#             input_dim=num_patches, output_dim=projection_dim
#         )
#
#     def call(self, patch):
#         positions = tf.range(start=0, limit=self.num_patches, delta=1)
#         encoded = self.projection(patch) + self.position_embedding(positions)
#         return encoded
#
# def patches(dropout,learning_rate, n_classes,fine_tune=0):
#     inputs = tf.keras.layers.Input(shape=(224, 224, 3))
#     patches = CreatePatches(patch_size=128)(inputs)
#     capa = tf.keras.models.Model(inputs, patches)
#     model = tf.keras.Sequential([
#             Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
#             MaxPool2D(pool_size=(2, 2), strides=2),
#             Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
#             MaxPool2D(pool_size=(2, 2), strides=2),
#             Flatten(),
#             Dense(units=n_classes, activation='softmax')
#     ])
#     #model.build(input_shape=(224, 224, 4))
#     model.add(capa)
#     model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
#     model.summary()
#     return model

def xception(dropout,learning_rate, n_classes,input_shape,fine_tune=0):
    xception_model = Xception(weights="imagenet", include_top=False)
    if fine_tune > 0:
        for layer in xception_model.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in xception_model.layers:
            layer.trainable = False
    # model = Sequential()
    # for layer in xception_model.layers[:-1]:
    #     model.add(layer)
    # for layer in model.layers:
    #     layer.trainable = False

    # if fine_tune > 0:
    #     for layer in model.layers[:-fine_tune]:
    #         layer.trainable = False
    # else:
    #     for layer in model.layers:
    #         layer.trainable = False

    # model.add(xception_model)
    # model.add(Dense(units=n_classes, activation='softmax'))

    # model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()

    #xception_model.trainable = False
    input = Input(shape=input_shape)
    model = xception_model(input)
    if dropout > -1:
        model = tf.keras.layers.Dropout(dropout)(model)
    model = tf.keras.layers.GlobalAveragePooling2D()(model)
    model = Dense(units=n_classes, activation='softmax') (model)
    model = Model(inputs=input, outputs=model)
    # for layer in xception_model.layers:
    #     layer.trainable = False
    return model