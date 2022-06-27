from image_model import mobile_net_model,xception,VGG16_model2#,conv_model,patches
from clinical_model import clinical_model
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
import tensorflow_addons as tfa


def build_image_model(dropout,learning_rate, n_classes,fine_tune,model_name,input_shape):
    if model_name == "VGG16":
        return VGG16_model2(dropout,learning_rate, n_classes,fine_tune,input_shape)
    elif model_name == "mobile_net":
        return mobile_net_model(dropout,learning_rate, n_classes,fine_tune,input_shape)
    # elif model_name == "conv":
    #     return conv_model(dropout,learning_rate, n_classes,fine_tune)
    # elif model_name == "patches":
    #     return patches(dropout,learning_rate, n_classes,fine_tune)
    elif model_name == "xception":
        return xception(dropout,learning_rate, n_classes,input_shape,fine_tune)
    else:
        raise ValueError("model_name must be one of VGG16, mobile_net, xception")


def build_model(dropout,learning_rate, n_classes,fine_tune,model_name,input_shape,depth,width,activation_function,image_model=True,clinical=False,clinical_num=0):
    if image_model:
        img_model = build_image_model(dropout,learning_rate, n_classes,fine_tune,model_name,input_shape)
    if clinical:
        clinic_model = clinical_model(clinical_num,n_classes,depth,width,activation_function)
    if image_model and clinical:
        combinedInput = concatenate([img_model.output, clinic_model.output])
        x = Dense(n_classes * 64, activation="relu")(combinedInput)
        x = Dense(n_classes * 64, activation="relu")(x)
        x = Dense(n_classes, activation="relu")(combinedInput)
        x = Dense(n_classes, activation="softmax")(x)

        model = tf.keras.models.Model(inputs=[img_model.input,clinic_model.input], outputs=x)
    elif image_model:
        model =  img_model
    elif clinical:
        model = clinic_model
    else:
        raise ValueError("No model selected")
    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tfa.metrics.F1Score(average='macro',num_classes=n_classes)])
    model.summary()
    return model


