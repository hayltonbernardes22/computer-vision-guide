# Image preprocessing Part - 1 

import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image
import cv2 

# %matplotlib inline 
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img

#Steps: 

# * 1- Loading image in RGB color 
# * 2- Convert image in array 
# * 3- Expand array dimensions (axis=0)  | passed to format <b>(Batch size, img height, img width, channels) </b>
# * 4- Scaling pixels between 0-1 (255)


image = "mountain.jpg"

def image_preprocess(image):

  img = load_img(image, color_mode="rgb")
  img = img_to_array(img)
  img = np.expand_dims(img, axis=0)
  img /= 255.0
  return img

image_preprocess(image)

# without array_to_img
image = "mountain.jpg"
image = load_img(image, target_size=(250, 250, 3))
image

# with array_to_img 
image = "mountain.jpg"
image = load_img(image)
image = array_to_img(image, scale=True)

plt.figure(figsize=(10,5))
plt.title("Mountain snow", fontsize=16)
plt.imshow(image, cmap="gray")
plt.tight_layout()
plt.show()

# Resize image 
dim = (250, 250)
image = "mountain.jpg"

img = cv2.imread(image)
img = cv2.resize(img, dim)
img = array_to_img(img)
img
