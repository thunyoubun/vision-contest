import pandas as pd
import os
import cv2
from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

PREDICT_DIR = "test images/test images"
FILENAME = 'filelist.txt'

df = pd.read_csv(FILENAME, delimiter=',', header=0)

def flatten(l):
    return [item for sublist in l for item in sublist]

IM_SIZE = 256
BATCH_SIZE = 100

datagen = ImageDataGenerator(rescale=1./255)

fileNames = flatten([df.columns.values.tolist()] + df.values.tolist())



model_test = load_model('newMenu.h5')
# score = model_test.evaluate_generator(
#     test_generator,
#     steps=len(test_generator))
# print('score (cross_entropy, accuracy):\n',score)

resultMap = {0: "B",
             1: "D",
             2: "R",
             3: "S"}
mapping = {('burger', 0), ('dimsum', 1), ('ramen', 2), ('sushi', 3)}
mapping = dict((v, k) for k, v in mapping)
PREDICT_IMAGE_SIZE = (IM_SIZE,IM_SIZE)


with open('result.txt', 'w') as writefile:
    for fileName in fileNames:
        test_im = cv2.imread(PREDICT_DIR + "/" + fileName, cv2.IMREAD_COLOR)
        test_im = cv2.resize(test_im, PREDICT_IMAGE_SIZE)
        #cv2.imshow("DD",test_im)
        test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
        test_im = test_im / 255.
        test_im = np.expand_dims(test_im, axis=0)
        w_pred = model_test.predict(test_im)
        
        print(w_pred)
        argMax = np.argmax(w_pred,axis = -1)[0]
        print(fileName + "::" + mapping[argMax])
        writefile.write(fileName + "::" + resultMap[argMax] + "\n")
        print()
    writefile.close()