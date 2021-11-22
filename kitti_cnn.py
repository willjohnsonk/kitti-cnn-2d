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

opt = keras.optimizers.Adam(lr = 0.0005)  # The learning rate details for the network

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu', input_shape=(30,30,1)))
model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
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



##### Existing example code for reference ################################################################################
# ### Defining variables, retrieving and augmenting training data ########################################################
# total_classes = 43
# cur_directory = os.getcwd()
# X = []    # Array for storing images
# Y = []    # Matching array for storing class labels for the given image. match indices

# for index in range(total_classes):
#     path = os.path.join(cur_directory, 'train', str(index))
#     images = os.listdir(path)
  
#     for img in images:
#         try:                                                  # image normalization via resize, color fix, equalizer
#             image = cv2.imread((os.path.join(path,img)))
#             image = cv2.resize(image, (30, 30))
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
#             image = np.array(image)
#             image = np.concatenate([np.expand_dims(cv2.equalizeHist(image[:,:,i]), axis=2) for i in range(3)], axis=2)
            
#             X.append(image)
#             Y.append(index)

#         except:
#             print("Error loading image")

# X = np.array(X)
# Y = np.array(Y)



# ### Creating and training the model ###################################################################################
# # Sets aside 20% of the original data to use for testing, all else is used to train. Randomizes.
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 

# y_train = to_categorical(y_train, 43)
# y_test = to_categorical(y_test, 43)

# opt = keras.optimizers.Adam(lr = 0.0008)  # The learning rate details for the network

# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(30,30,3)))
# model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(rate=0.25))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(rate=0.25))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(rate=0.5))
# model.add(Dense(43, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# epochs = 18   # How many iterations the model will run
# history = model.fit(x_train, y_train, batch_size=64, epochs=epochs, validation_data=(x_test, y_test)) # Saves metadata for later
# model.save('signCNN.h5')


# ### Model testing via sklearn metrics #################################################################################
# y_test = pd.read_csv('Test.csv')
# labels = y_test["ClassId"].values
# img_paths = y_test["Path"].values
# test_data=[]

# for path in img_paths:        # When making comparisons for metrics you must do the same augmentation as for training
#     image = cv2.imread(path)
#     image = cv2.resize(image, (30, 30))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = np.array(image)
#     image = np.concatenate([np.expand_dims(cv2.equalizeHist(image[:,:,i]), axis=2) for i in range(3)], axis=2)
    
#     test_data.append(image)

# test_data = np.array(test_data)
# pred = np.argmax(model.predict(test_data), axis=-1)
# print("SKlearn accuracy metric: ", accuracy_score(labels, pred))


# ### Defining accuraccy of the model on Polish street view images ######################################################
# expected = []
# guessed = []
# confidence = []
# pol_class = []
# pol_data_path = '/home/peyton/tfCode/signCNN/polSign/imgs'
# pol = os.listdir(pol_data_path)
# err_count = 0
# count = 0
# total = 0

# # Used to estimate what class some test values are in and at what confidence for end result
# for img in pol:  
#     try:
#         pol_class = str(os.path.splitext(img)[0])
#         pol_class = int(pol_class[:2])
#         expected.append(pol_class)

#         p_img = cv2.imread(os.path.join(pol_data_path, img))
#         p_img = cv2.resize(p_img, (30, 30))
#         p_img = cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB)
#         p_img = np.concatenate([np.expand_dims(cv2.equalizeHist(p_img[:,:,i]), axis=2) for i in range(3)], axis=2)
#         p_img = np.expand_dims(p_img, axis=0)

#         predicted = model.predict(p_img)
#         guessed.append(np.argmax(predicted[0]))
#         confidence.append(np.amax(predicted))

#     except:
#         print("Could not estimage average POL confidence")

# for i in range(len(expected)):
#     if(expected[i] != guessed[i]):
#         err_count = err_count + 1
    
#     if(expected[i] == guessed[i]):
#         count = count + 1
#         total = total + confidence[i]

# pol_accurracy = (total / count) * (1 - (err_count / 45)) # math
# print("Incorrect classifications on POL: ", err_count)
# print("POL accuracy metric: ", pol_accurracy)



# ### Comparing the model's performance between sign metadata #############################################################
# expected_pol = []
# guessed_pol = []
# confidence_pol = []
# pol_path = '/home/peyton/tfCode/signCNN/polSign/meta'
# pol_imgs = os.listdir(pol_path)

# expected_deu = []
# guessed_deu = []
# confidence_deu = []
# deu_path = '/home/peyton/tfCode/signCNN/meta/comp'
# deu_imgs = os.listdir(deu_path)

# # Defining Polish metadata (same statistic setup, different day. mostly related to my actual project, less important)
# for img in pol_imgs: # these two image sets should be converted to nparray for readability and dataframes
#     try:
#         img_class = int(os.path.splitext(img)[0])
#         expected_pol.append(img_class)

#         pol_img = cv2.imread(os.path.join(pol_path,img))
#         pol_img = cv2.resize(pol_img, (30, 30))
#         pol_img = cv2.cvtColor(pol_img, cv2.COLOR_BGR2RGB)
#         pol_img = np.concatenate([np.expand_dims(cv2.equalizeHist(pol_img[:,:,i]), axis=2) for i in range(3)], axis=2)
#         pol_img = np.expand_dims(pol_img, axis=0)

#         predict_pol = model.predict(pol_img)
#         guessed_pol.append(np.argmax(predict_pol[0]))
#         confidence_pol.append(100 * (np.amax(predict_pol)))

#     except:
#         print("Could not estimate class for POL img")

# # Defining German metadata
# for img in deu_imgs:
#     try:
#         img_class = int(os.path.splitext(img)[0])
#         expected_deu.append(img_class)

#         deu_img = cv2.imread(os.path.join(deu_path,img))
#         deu_img = cv2.resize(deu_img, (30, 30))
#         deu_img = cv2.cvtColor(deu_img, cv2.COLOR_BGR2RGB)
#         deu_img = np.concatenate([np.expand_dims(cv2.equalizeHist(deu_img[:,:,i]), axis=2) for i in range(3)], axis=2)
#         deu_img = np.expand_dims(deu_img, axis=0)

#         predict_deu = model.predict(deu_img)
#         guessed_deu.append(np.argmax(predict_deu[0]))
#         confidence_deu.append(100 * (np.amax(predict_deu)))

#     except:
#         print("Could not estimate class for DEU img")

# # Creating dataframes using panda
# data_pol = {
#     "Expected": expected_pol,
#     "Guessed": guessed_pol,
#     "Confidence": confidence_pol
# }
# data_deu = {
#     "Expected": expected_deu,
#     "Guessed": guessed_deu,
#     "Confidence": confidence_deu
# }

# df_pol = pd.DataFrame(data_pol)
# df_deu = pd.DataFrame(data_deu)
# # df.sort_values(by=['Expected'], inplace=True)

# print("Polish Results")
# print(df_pol)
# print("German Results")
# print(df_deu)



# ### Model performance graphs ############################################################################################
# plt.figure(0)
# plt.plot(history.history['accuracy'], label='training accuracy')  # Here is where the logged history info from the model is used
# plt.plot(history.history['val_accuracy'], label='val accuracy')
# plt.title('Accuracy')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.legend()

# plt.figure(1)
# plt.plot(history.history['loss'], label='training loss')
# plt.plot(history.history['val_loss'], label='val loss')
# plt.title('Loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend()

# plt.show()





# ### Leftover test code for comparing augmented images (visualization junk)
# f = plt.figure(figsize=(6,4))

# path = '/home/peyton/tfCode/signCNN/train/1/00001_00012_00000.png'
# img = cv2.imread(path, cv2.IMREAD_COLOR)
# img = cv2.resize(img, (30, 30))
# sp = f.add_subplot(1,3,1)
# plt.imshow(img)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# sp = f.add_subplot(1,3,2)
# plt.imshow(img)

# img = np.concatenate([np.expand_dims(cv2.equalizeHist(img[:,:,i]), axis=2) for i in range(3)], axis=2)
# sp = f.add_subplot(1,3,3)
# plt.imshow(img)

# plt.show()

# # Old data for reading/modifying images with cv2
# # Saves all the training images to an array
# for img in images:
#     try:
#         image = cv2.imread((os.path.join(train_img_path,img)))
#         image = np.array(image)
#         X.append(image)

#     except:
#         print("Error loading image")

# # Example image stuff
# img_path = os.path.join(cur_dir, train_img_dir, "000000.png")
# image = cv2.imread(img_path)
# image = cv2.rectangle(image, (712, 143), (810, 307), (255,0,0), 2)  # Draws a bounding box. Requires int type
#                                                                     # first point is top left, second is bottom right,
#                                                                     # then color, then line width.

# cv2.imshow('Test Image', image) # Displays image in a window. Press any key to exit.
# cv2.waitKey(0)
# cv2.destroyAllWindows()
