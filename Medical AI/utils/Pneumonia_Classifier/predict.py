import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from PIL import Image
from werkzeug.datastructures import FileStorage
import io
import base64

# Defined CNN model
def load_model():
    cnn = Sequential()
    cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(32, (3, 3), activation="relu"))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(activation = 'relu', units = 128))
    cnn.add(Dense(activation = 'sigmoid', units = 1))
    cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # print(cnn.summary())
    return cnn


# for predicting images for pneumonia
def predict(image):

    classes = ['Normal','Pneumonia']

    # getting imahe from web app as FileStorage obj
    org_image=None
    if image and isinstance(image, FileStorage):
        image = Image.open(image)
        org_image=image
    else:
        return("")

    # loads CNN architecture defined above
    model = load_model()

    # loads pretrained weights for CNN moel
    model.load_weights("utils/Pneumonia_Classifier/pneumonia_detection_model.h5")

    # print(image)
    #preprocessing image
    image = image.resize((64, 64)).convert("RGB")
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)

    # predicting image
    prediction = model.predict(image)

    # sending image as base64 bytes array as str
    img_byte_arr = io.BytesIO()
    org_image.save(img_byte_arr, format='JPEG')
    result_image = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')

    return(str(int(prediction[0][0])),result_image)

