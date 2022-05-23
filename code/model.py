from image_model import VGG16_model,mobile_net_model,conv_model,patches,xception,VGG16_model2
from clinical_model import clinical_model
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras import optimizers


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

def build_model(learning_rate, n_classes,fine_tune,model_name,input_shape,image_model=True,clinical=False,clinical_num=0):
    if image_model:
        img_model = build_image_model(learning_rate, n_classes,fine_tune,model_name,input_shape)
    if clinical:
        clinic_model = clinical_model(clinical_num,n_classes)
    if image_model and clinical:
        combinedInput = concatenate([img_model.output, clinic_model.output])
        x = Dense(n_classes, activation="relu")(combinedInput)
        x = Dense(n_classes, activation="softmax")(x)
        model = tf.keras.models.Model(inputs=[img_model.input,clinic_model.input], outputs=x)
    elif image_model:
        model =  img_model
    elif clinical_model:
        model = clinic_model
    #optimizer = optimizers.Adam(clipvalue=0.5)
    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate,clipvalue=0.5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


