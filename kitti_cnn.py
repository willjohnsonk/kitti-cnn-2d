### Relevant info via the official readme ################################################################################
# Link to data (requires account) http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d

# Despite the fact that we have labeled 8 different classes, only the
# classes 'Car' and 'Pedestrian' are evaluated in our benchmark, as only for
# those classes enough instances for a comprehensive evaluation have been
# labeled

# DontCare' labels denote regions in which objects have not been labeled,
# for example because they have been too far away from the laser scanner. To
# prevent such objects from being counted as false positives our evaluation
# script will ignore objects detected in don't care regions of the test set.
# You can use the don't care labels in the training set to avoid that your object
# detector is harvesting hard negatives from those areas, in case you consider
# non-object regions from the training images as negative examples.

# 2D Object Detection Benchmark
# =============================

# The goal in the 2D object detection task is to train object detectors for the
# classes 'Car', 'Pedestrian', and 'Cyclist'. The object detectors must
# provide as output the 2D 0-based bounding box in the image using the format
# specified above, as well as a detection score, indicating the confidence
# in the detection. All other values must be set to their default values
# (=invalid), see above. One text file per image must be provided in a zip
# archive, where each file can contain many detections, depending on the 
# number of objects per image. In our evaluation we only evaluate detections/
# objects larger than 25 pixel (height) in the image and do not count 'Van' as
# false positives for 'Car' or 'Sitting Person' as false positive for 'Pedestrian'
# due to their similarity in appearance. As evaluation criterion we follow
# PASCAL and require the intersection-over-union of bounding boxes to be
# larger than 50% for an object to be detected correctly.

import os
from matplotlib import patches
import numpy as np
from numpy.core.fromnumeric import _cumprod_dispatcher
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout


### Metadata for labels ##################################################################################################
# Classes: 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare'
# Total classes: 7
# Label data (by index):
#   0           string - class/label name                               (required)
#   1           float - describing if the object is in frame
#   2           integer - describes object occlusion from 0 to 3
#   3           float - observation angle
#   4,5,6,7     float - 2d bounding box in pixel coordinates            (required)
#   8,9,10      float - object dimensions in 3D
#   11,12,13    float - object location in 3D
#   14          float - rotation of object around y axis
#   15          float - score (unused)

# Training Image Location: /data_object_image_2/training/image_2
# Testing Image Location:/data_object_image_2/testing/image_2
# Label Location: /data_object_label_2/training/label_2

# Image name format: "000000.png"
# Label name format: "000000.txt"

# https://stackoverflow.com/questions/7422204/intensity-normalization-of-image-using-pythonpil-speed-issues
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

### Defining variables and loading training data #########################################################################
# print(tf.test.gpu_device_name())  # For testing GPU accessibility on linux

# Xs: arrays for storing img data Y: array for storing int class labels (0-6)
X1 = []
X2 = []
X = []
Y = []
labelData = []

# Directory setup
cur_dir = os.getcwd()
train_img_dir = "data_object_image_2/training/image_2"
test_img_dir = "data_object_image_2/testing/image_2"
label_dir = "data_object_label_2/training/label_2"

# Gets the paths for the individual images in the training data
train_img_path = os.path.join(cur_dir, train_img_dir)
images = os.listdir(train_img_path)

# Gets the paths for the training labels
label_path = os.path.join(cur_dir, label_dir)
labels = os.listdir(label_path)

# Saves the label class and bounding box coords from each file
for data in labels:
    try:
        lines = []
        with open(os.path.join(label_path,data)) as f:
            lines = f.readlines()

        for line in lines:
            inst = line.split()
            if inst[0] != "DontCare" and inst[0] != "Misc" and int(inst[2]) < 1:
                # [dataframe, class, top, left, bottom right] for each index
                labelData.append([(str(data).rsplit('.',1)[0]), inst[0], round(float(inst[4])), round(float(inst[5])), round(float(inst[6])), round(float(inst[7]))])
    
    except:
        print("Error loading label")


# Crossreferences the image for each label and appends to X, augments for model
for i in range(len(labelData)):
    try:
        # Image filepath setup
        inst = labelData[i]
        img_file = inst[0] + '.png'
        directory = os.path.join(train_img_path,img_file)

        # Image saving/augmentation
        img = Image.open(directory).convert('L')                    # Coverts to greyscale. Remove ".convert('L') for RGB"
        img_res = img.crop((inst[2], inst[3], inst[4], inst[5]))    # Crops KITTI img to bounding box
        shape = (30, 30)                                            # Resizes img to 30 pixel square
        img_res = img_res.resize(shape)
        img_mirror = ImageOps.mirror(img_res)                       # Creates an duplicate image that's mirrored

        img_res = np.array(img_res)                                 # Convert to np array and save
        img_mirror = np.array(img_mirror)

        img_res = normalize(img_res)
        img_mirror = normalize(img_mirror)

        X1.append(img_res)
        X2.append(img_mirror)
        

    except:
        print("Error loading image with data ", inst, i)


# Saves associated labels to Y as ints (0-6)
label_dict = {'Car':0,'Van':1,'Truck':2,'Pedestrian':3,'Person_sitting':4,'Cyclist':5,'Tram':6}
for label in labelData:
    Y.append(label_dict[label[1]])

# Combines the mirrored and nonmirrored datasets
X = X1 + X2
Y = Y + Y

X = np.array(X)
Y = np.array(Y)

print(np.shape(X))
print(np.shape(Y))


### Creating and training the model ###################################################################################
# Sets aside 20% of the original data to use for testing, all else is used to train. Randomizes based on seed (42 here).
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Greyscale conversion removes dimension, add back filler to fit model
# If using RGB comment these two lines out and change input shape to (x, x, 3)
x_train = tf.expand_dims(x_train, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)

y_train = to_categorical(y_train, 7)
y_test = to_categorical(y_test, 7)

print(len(y_train))
print(len(y_test))

opt = keras.optimizers.Adam(lr = 0.0001)  # The learning rate details for the network

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(30,30,1)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.4))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

epochs = 100   # How many iterations the model will run
history = model.fit(x_train, y_train, batch_size=64, epochs=epochs, validation_data=(x_test, y_test)) # Saves metadata for later
model.save('kitti_cnn.h5')

### Model performance graphs ############################################################################################
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')  # Here is where the logged history info from the model is used
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.show()
