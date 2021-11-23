import os
from matplotlib import patches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

### Loading the trained model
try: 
    model = load_model('kitti_cnn.h5')
    print("Model successfully loaded")
except:
    print("Could not load model")

### Initializing arrays for data and directories
cur_dir = os.getcwd()
test_img_dir = 'data_object_image_2/testing/crop_image_2'
test_imgs_path = os.path.join(cur_dir, test_img_dir)
test_imgs = os.listdir(test_imgs_path)

tests = []
names = []
prediction = []
label_guess = []
label_conf = []

### Iterates over cropped test images, applies augmentations
for img in test_imgs:
    directory = os.path.join(test_imgs_path, img)
    names.append(img)

    test_img = Image.open(directory).convert('L')   # Coverts to greyscale. Remove ".convert('L') for RGB"
    test_img = test_img.resize((30, 30))            # Resizes img to 30 pixel square
    test_img = np.array(test_img)                   # Convert to np array and save
    test_img = normalize(test_img)                  # Applies intensity normalization on pixels

    test_img = np.expand_dims(test_img, axis=0)
    tests.append(test_img)

tests = np.expand_dims(tests, axis=-1)

for i in range(len(tests)):
    prediction = model.predict(tests[i])
    label_guess.append(np.argmax(prediction[0]))
    label_conf.append(100 * (np.amax(prediction)))

    print("Guessed: ", np.argmax(prediction[0]), names[i])