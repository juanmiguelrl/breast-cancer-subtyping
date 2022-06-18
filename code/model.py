from image_model import VGG16_model,mobile_net_model,xception,VGG16_model2#,conv_model,patches
from clinical_model import clinical_model
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras import optimizers
import tensorflow_addons as tfa

# from tensorflow.keras import backend as K
#
# def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
#
# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision
#
# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))

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
        raise ValueError("model_name must be one of VGG16, mobile_net, conv, patches, xception")
        #return VGG16_model2(learning_rate, n_classes,fine_tune,input_shape)


def build_model(dropout,learning_rate, n_classes,fine_tune,model_name,input_shape,depth,width,activation_function,image_model=True,clinical=False,clinical_num=0):
    if image_model:
        img_model = build_image_model(dropout,learning_rate, n_classes,fine_tune,model_name,input_shape)
    if clinical:
        clinic_model = clinical_model(clinical_num,n_classes,depth,width,activation_function)
    if image_model and clinical:
        combinedInput = concatenate([img_model.output, clinic_model.output])
        x = Dense(n_classes, activation="relu")(combinedInput)
        x = Dense(n_classes, activation="softmax")(x)
        model = tf.keras.models.Model(inputs=[img_model.input,clinic_model.input], outputs=x)
    elif image_model:
        model =  img_model
    elif clinical:
        model = clinic_model
    else:
        raise ValueError("No model selected")
    #optimizer = optimizers.Adam(clipvalue=0.5)
    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tfa.metrics.F1Score(average='macro',num_classes=n_classes)])
                  # metrics=['accuracy',f1_m,precision_m, recall_m])
    model.summary()
    return model


