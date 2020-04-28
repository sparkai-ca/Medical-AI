import numpy as np
from utils.MaskRCNN.mrcnn import utils
from utils.MaskRCNN.mrcnn import visualize
from PIL import Image
import tensorflow as tf
from utils.MaskRCNN.mrcnn.config import Config
from utils.MaskRCNN.mrcnn import model as modellib, utils
from werkzeug.datastructures import FileStorage
import time
import cv2
import io
import base64

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

############################################################
#  Configurations
############################################################

class GenericConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """

    def __init__(self, classes, steps):
        self.NUM_CLASSES = classes
        self.STEPS_PER_EPOCH = steps
        super().__init__()

    # Give the configuration a recognizable name
    NAME = "class"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.75
    IMAGE_MAX_DIM = 448
    IMAGE_MIN_DIM = 384
    TRAIN_ROIS_PER_IMAGE = 20
    DETECTION_NMS_THRESHOLD = 0.75
    DETECTION_MAX_INSTANCES = 300


def predict(image):
	# getting image from web app as FileStorage
	if image and isinstance(image, FileStorage):
		image = Image.open(image)
		image = np.array(image)
	else:
		return("")

	weights_path = "utils/MaskRCNN/w8s_20.h5"
	MODEL_DIR = "/".join(weights_path.split("/")[:-2])

	config = None
	model=None
	r=None

	# configuring model
	config = GenericConfig(4, 1)

	# Create model in inference mode
	with tf.device("/cpu:0"):
		model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
	model.load_weights(weights_path, by_name=True)

	labels = ["RBC","WBC","Platelets"]

	# detecting from image for predictions
	results = model.detect([image], verbose=0)

	r = results[0]

	classes = ["background"]
	classes += labels

	# getting PIL image of detected cells types rois
	result = visualize.process_image(image,r['rois'],None,r['class_ids'],r['scores'],classes,save_dir='static')

	result_str = result[0]
	result_image = result[1]

	#sending image to web app as bytes array os str
	img_byte_arr = io.BytesIO()
	result_image.save(img_byte_arr, format='JPEG')
	result_image = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')


	config = None
	model=None
	r=None
	image=None


	return(result_str,result_image)





"""
Mask R-CNN Prediction.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Originally Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 main.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 main.py train --dataset=/path/to/balloon/dataset --weights=last


"""



