from image_model import VGG16_model,mobile_net_model,conv_model,patches,xception,VGG16_model2
from clinical_model import clinical_model
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate


def build_image_model(learning_rate, n_classes,fine_tune,model_name,input_shape):
    if model_name == "vgg16":
        return VGG16_model(learning_rate, n_classes,fine_tune,input_shape)
    elif model_name == "mobile_net":
        return mobile_net_model(learning_rate, n_classes,fine_tune)
    elif model_name == "conv":
        return conv_model(learning_rate, n_classes,fine_tune)
    elif model_name == "patches":
        return patches(learning_rate, n_classes,fine_tune)
    elif model_name == "xception":
        return xception(learning_rate, n_classes,fine_tune)
    else:
        return VGG16_model2(learning_rate, n_classes,fine_tune,input_shape)

def build_model(learning_rate, n_classes,fine_tune,model_name,input_shape,image_model=True,clinical_model=False,clinical_num=0):
    if image_model:
        img_model = build_image_model(learning_rate, n_classes,fine_tune,model_name,input_shape)
    if clinical_model:
        clinic_model = clinical_model(clinical_num,n_classes)
    if image_model and clinical_model:
        combinedInput = concatenate([img_model.output, clinic_model.output])
        x = Dense(n_classes, activation="relu")(combinedInput)
        x = Dense(n_classes, activation="linear")(x)
        model = tf.keras.models.Model(inputs=[img_model.input,clinic_model.input], outputs=clinic_model(img_model.output))
        return model
    elif image_model:
        return img_model
    elif clinical_model:
        return clinic_model
