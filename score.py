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

model_test = load_model('newMenu2.h5')
TARGET_SIZE = 256
BATCH_SIZE = 50

mapping = {('burger', 0), ('dimsum', 1), ('ramen', 2), ('sushi', 3)}
mapping = dict((v, k) for k, v in mapping)

datagen = ImageDataGenerator(rescale=1./255)
test_generator = datagen.flow_from_directory(
    'test_images',
    shuffle=False,
    target_size=(TARGET_SIZE,TARGET_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    )

score = model_test.evaluate(
    test_generator,
    steps=len(test_generator))
print('score (cross_entropy, accuracy):\n',score)

# test_generator.reset()
# predict = model_test.predict(
#     test_generator,
#     steps=len(test_generator),
#     workers = 1,
#     use_multiprocessing=False)

# # print("predict :", predict)

# predict_class_idx = np.argmax(predict,axis = -1)
# # print('predicted class index:\n', predict_class_idx)

# mapping = dict((v,k) for k,v in test_generator.class_indices.items())
# predict_class_name = [mapping[x] for x in predict_class_idx]
# # print('predicted class name:\n', predict_class_name)

# resultArr = []
# menus = ["Burger", "Dimsum","Ramen","Sushi"]
# src_test = "test_images"

# for i in range(len(menus)) : # menu ทั้ง 4
#     img_path = src_test + "/" + menus[i] # set path ของ menu
#     arr = os.listdir(img_path)
#     for j in range(len(arr)): # 100 รูป
#         class_img = predict_class_name[i*100+ j]
#         resultArr.append(arr[j-1] + "::" + class_img[0]  + "\n")


# file = open("result.txt", "w+")
# content = str(resultArr)
# for name in resultArr:
#     file.writelines(name)
# file.close()

# cm = confusion_matrix(test_generator.classes, np.argmax(predict,axis = -1))
# print("Confusion Matrix:\n",cm)


# test_im = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
# im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
# im_resized = cv2.resize(im, (TARGET_SIZE, TARGET_SIZE))
# im = im_resized/255.
# im_reshape = im.reshape(1, TARGET_SIZE, TARGET_SIZE, 3)

# w_pred = model_test.predict(im_reshape)
    
# print(w_pred)

# predict_class_idx = np.argmax(w_pred, axis=-1)[0]
# predict_class_name = mapping[predict_class_idx]

# text = "89829495_497038170969451_889396434597164704_n.jpg" + "::" + predict_class_name[0].upper() + '\n'
# print(text)
