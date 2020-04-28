import skimage.transform as trans
import numpy as np
from utils.Tumor_detection.model import unet
import cv2
from werkzeug.datastructures import FileStorage
from skimage import img_as_ubyte
from PIL import Image
import io
import base64

target_size = (256,256)

# This function preprocesses the image for inference

def preprocess(img):
    target_size = (256,256)
    img = trans.resize(img, (target_size[0],target_size[1],1))
    img = np.reshape(img, (target_size[0],target_size[1],1)) if (not False) else img
    img = np.reshape(img, (1,) + img.shape)
    return img

# this function is used to get image segmentation mask from unet
def get_mask(results):
    for i, item in enumerate(results):
        img = item[:, :, 0]
        return img

# converts image to desired traget 
def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()
    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

# this will return largest rois consists mask of Detected Tumor 
def find_largest_contour(image, orig_image):
    orig_image = cv2.cvtColor(orig_image[:,:,:1],cv2.COLOR_GRAY2RGB)
    gray = convert(image, 0, 255, np.uint8)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) > 2000:
            cv2.rectangle(orig_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return orig_image
        else:
            #  No tumor
            return None


def predict(image):
    # getting image from webbapp as FileStorage
    if image and isinstance(image, FileStorage):
        image = Image.open(image)
        image = np.array(image)
    else:
        return("")

    # preprocessing and predicting image and getting an roi with tumor detection
    model = unet(pretrained_weights="utils/Tumor_detection/unet3_w8.hdf5")
    width,height,channels = image.shape
    preprocessed_image = preprocess(image)
    results = model.predict(preprocessed_image)
    mask = get_mask(results)
    mask = img_as_ubyte(mask)
    cv_image = img_as_ubyte(image)
    mask = cv2.resize(mask,(width,height))
    ret, gray = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    image = find_largest_contour(mask,image)

    # sending image to webapp
    result_image = None
    if not image is None:
        img_byte_arr = io.BytesIO()
        Image.fromarray(image).save(img_byte_arr, format='JPEG')
        result_image = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    else:
        return("")

    return("_/",result_image)
