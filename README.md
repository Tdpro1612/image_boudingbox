# Cat Face Localization

* đã có data xem file Cat Face Localization.ipynb
* chưa có data dowload file bằng file API dowload data từ Kaggle.ipynb

# hướng dẫn cách làm
we using google colab
###cài đặt thư viện
```
import os
import random

import cv2

import numpy as np
import pandas as pd
import tensorflow as tf

from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
```
### Build your own model
```
ở đây xài Densenet121 nếu muốn có thể xài mạng khác để thử
base_model = tf.keras.applications.DenseNet121(include_top=False,weights="imagenet",input_shape=(224,224,3))
base_model.trainable = False
model = tf.keras.Sequential([base_model,tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(4,activation='softmax')])

# Print out model summary
model.summary()
```
![train model](https://user-images.githubusercontent.com/61773507/96671767-d4b65380-138c-11eb-969b-17783c7c43f6.png)

#test
