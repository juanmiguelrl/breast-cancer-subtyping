import tensorflow as tf
from tensorflow.keras.applications import VGG16


def VGG16_model(learning_rate, n_classes):
    # Load VGG16 trained params and CNN network
    pre_trained_model = VGG16(input_shape=(224, 224, 3),
                              include_top=False,
                              weights='imagenet')
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

    # 2 full conected layers to be trained are added to the model
    model = tf.keras.models.Sequential([pre_trained_model,
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(256, activation='relu'),
                                        tf.keras.layers.Dense(n_classes, activation='softmax')
                                        ])
    model.summary()

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                  metrics=['acc'])
    return model