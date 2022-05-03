import tensorflow as tf
from tensorflow.keras.applications import VGG16


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
