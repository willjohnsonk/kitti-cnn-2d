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


img_name = []

for i in range(len(tests)):
    img_name.append(names[i])
    prediction = model.predict(tests[i])
    label_guess.append(np.argmax(prediction[0]))
    label_conf.append(100 * (np.amax(prediction)))

label_dict = {0:'Car',1:'Van',2:'Truck',3:'Pedestrian',4:'Person_sitting',5:'Cyclist',6:'Tram'}

id_dict = {
    '000000.png':'Pedestrian', '000001.png':'Car', '000002.png':'Cyclist', '000003.png':'Pedestrian',
    '000004.png':'Car', '000005.png':'Car', '000006.png':'Truck', '000007.png':'Car',
    '000008.png':'Van', '000009.png':'Truck', '000010.png':'Van', '000011.png':'Tram',
   '000012.png':'Pedestrian', '000013.png':'Cyclist', '000014.png':'Cyclist', '000015.png':'Truck',
   '000016.png':'Cyclist', '000017.png':'Car', '000018.png':'Car', '000019.png':'Tram'
}

guessed = []
expected = []

for label in label_guess:
    guessed.append(label_dict[label])

for exp in img_name:
    expected.append(id_dict[exp])

test_data = {
    "Image ID": img_name,
    "Expected": expected,
    "Guessed": guessed,
    "Confidence": label_conf
}

test_frame = pd.DataFrame(test_data)
test_frame.sort_values(by=['Confidence'], inplace=True, ascending=False)

print(test_frame)
