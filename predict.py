import pandas as pd
import os
import cv2
from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

model_test = load_model('menuset.h5')
TARGET_SIZE = 128
mapping = {('burger', 0), ('dimsum', 1), ('ramen', 2), ('sushi', 3)}
mapping = dict((v, k) for k, v in mapping)
IMG_PATH = "test images/test images/334965556_6930039340345845_7981709598339004335_n.jpg"
test_im = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
im_resized = cv2.resize(im, (TARGET_SIZE, TARGET_SIZE))
im = im_resized/255.
im_reshape = im.reshape(1, TARGET_SIZE, TARGET_SIZE, 3)
score = model_test.evaluate(im_reshape)
print(score)
w_pred = model_test.predict(im_reshape)
    
print(w_pred)

predict_class_idx = np.argmax(w_pred, axis=-1)[0]
predict_class_name = mapping[predict_class_idx]

text = "89829495_497038170969451_889396434597164704_n.jpg" + "::" + predict_class_name[0].upper() + '\n'
print(text)
