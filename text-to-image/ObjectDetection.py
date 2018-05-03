
import keras
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import tensorflow as tf
from collections import Counter

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

#Set the modified TF session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

class ImageDetect:
    
    def __init__(self, path):

        #Load label to names mapping for visualization purposes
        self.labels_dict = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
                       5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
                       10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 
                       14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
                       20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 
                       25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
                       30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
                       35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 
                       39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
                       45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
                       51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 
                       57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 
                       62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 
                       68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 
                       73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 
                       77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

        #Load retinanet model
        self.model = keras.models.load_model(path, custom_objects=custom_objects)
    
    def predict(self, image):

        #Preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        #Process image
        start = time.time()
        _, _, detections = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        print("Processing Time : ", time.time() - start)

        #Compute predicted labels and scores
        predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
        scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]

        #Correct for image scale
        detections[0, :, :4] /= scale

        #Get Predictions
        final = dict([(i[1],0) for i in self.labels_dict.items()])
        for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
            if score < 0.5:
                continue
            final[self.labels_dict[label]] += 1
        return(Counter(final))