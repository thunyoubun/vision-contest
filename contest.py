#For Goole Colab Version
#https://colab.research.google.com/drive/138XnTYRSe4HIg_XELX-RixCVHuxDQHao?usp=share_link

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


IM_SIZE = 128
BATCH_SIZE = 100

#Create model
input = Input(shape = (IM_SIZE,IM_SIZE,3))
conv1 = Conv2D(32,3,activation='relu')(input)
pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64,3,activation='relu')(pool1)
pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(128,3,activation='relu')(pool2)
pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
flat = Flatten()(pool3)
hidden = Dense(16, activation='relu')(flat)
output = Dense(4, activation='softmax')(hidden)
model = Model(inputs=input, outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()



datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    'Dataset/train',
    shuffle=True,
    target_size=(IM_SIZE,IM_SIZE),
    batch_size=BATCH_SIZE,
    color_mode = 'rgb',
    class_mode='categorical')

validation_generator = datagen.flow_from_directory(
    'Dataset/validation',
    shuffle=False,
    target_size=(IM_SIZE,IM_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical')

test_generator = datagen.flow_from_directory(
    'Dataset/test',
    shuffle=False,
    target_size=(IM_SIZE,IM_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical')



#Train Model
# checkpoint = ModelCheckpoint('Dataset.h5', verbose=1, monitor='val_accuracy',save_best_only=True, mode='max')

# h = model.fit_generator(
#     train_generator,
#     epochs=20,
#     steps_per_epoch=len(train_generator),
#     validation_data=validation_generator,
#     validation_steps=len(validation_generator),
#     callbacks=[checkpoint])

# plt.plot(h.history['accuracy'])
# plt.plot(h.history['val_accuracy'])
# plt.legend(['train', 'val'])



#Test Model
model = load_model('Dataset.h5')
score = model.evaluate_generator(
    test_generator,
    steps=len(test_generator))
print('score (cross_entropy, accuracy):\n',score)


test_generator.reset()
predict = model.predict_generator(
    test_generator,
    steps=len(test_generator),
    workers = 1,
    use_multiprocessing=False)
print('confidence:\n', predict)

predict_class_idx = np.argmax(predict,axis = -1)
print('predicted class index:\n', predict_class_idx)

mapping = dict((v,k) for k,v in test_generator.class_indices.items())
predict_class_name = [mapping[x] for x in predict_class_idx]
print('predicted class name:\n', predict_class_name)


resultArr = []
menus = ["Burger", "Dimsum","Ramen","Sushi"]
src_test = "C:/Users/thun_/Desktop/vision-contest/Dataset/test/"


for i in range(len(menus)) : # menu ทั้ง 4
    img_path = src_test + menus[i] # set path ของ menu
    arr = os.listdir(img_path)
    for j in range(len(arr)): # 100 รูป
        class_img = predict_class_name[i*len(arr) + j]
        resultArr.append(arr[j-1] + "::" + class_img[0]  + "\n")


file = open("result.txt", "w+")
content = str(resultArr)
for name in resultArr:
    file.writelines(name)
file.close()



cm = confusion_matrix(test_generator.classes, np.argmax(predict,axis = -1))
print("Confusion Matrix:\n",cm)

plt.show()



