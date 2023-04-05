import os
import cv2
from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Input , Dropout
from keras.layers import Conv2D, MaxPool2D ,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


IM_SIZE = 256
BATCH_SIZE = 50

#Create model
input = Input(shape = (IM_SIZE,IM_SIZE,3))
conv1 = Conv2D(32,3,activation='relu')(input)
conv1 = Conv2D(32,3,activation='relu')(conv1)
conv1 = BatchNormalization()(conv1)
pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64,3,activation='relu')(pool1)
conv2 = Conv2D(64,3,activation='relu')(conv2)
conv2 = BatchNormalization()(conv2)
pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(128,3,activation='relu')(pool2)
conv3 = Conv2D(128,3,activation='relu')(conv3)
conv3 = BatchNormalization()(conv3)
pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(256,3,activation='relu')(pool3)
conv4 = Conv2D(256,3,activation='relu')(conv4)
conv4 = BatchNormalization()(conv4)
pool4 = MaxPool2D(pool_size=(2, 2))(conv4)
conv5 = Conv2D(512,3,activation='relu')(pool4)
conv5 = Conv2D(512,3,activation='relu')(conv5)
conv5 = BatchNormalization()(conv5)
pool5 = MaxPool2D(pool_size=(2, 2))(conv5)
conv6 = Conv2D(32,3,activation='relu')(pool5)
conv6 = Conv2D(32,3,activation='relu')(conv6)
conv6 = BatchNormalization()(conv6)
pool6 = MaxPool2D(pool_size=(2, 2))(conv6)
flat = Flatten()(pool6)
dense1 = Dense(128, activation='relu')(flat)
dense1 = Dropout(0.5)(dense1)
dense2 = Dense(64, activation='relu')(dense1)
dense2 = Dropout(0.5)(dense2)
dense3 = Dense(32, activation='relu')(dense2)
dense3 = Dropout(0.5)(dense3)
output = Dense(4, activation='softmax')(dense3)
model = Model(inputs=input, outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()



datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    'Dataset2/train',
    shuffle=True,
    target_size=(IM_SIZE,IM_SIZE),
    batch_size=BATCH_SIZE,
    color_mode = 'rgb',
    class_mode='categorical')

validation_generator = datagen.flow_from_directory(
    'Dataset2/validation',
    shuffle=False,
    target_size=(IM_SIZE,IM_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical')

test_generator = datagen.flow_from_directory(
    'Dataset2/test',
    shuffle=False,
    target_size=(IM_SIZE,IM_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical')




#Train Model
checkpoint = ModelCheckpoint('newMenu3.h5', verbose=1, monitor='val_accuracy',save_best_only=True, mode='max')

h = model.fit(
    train_generator,
    epochs=20,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint])

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.legend(['train', 'val'])



#Test Model
model = load_model('newMenu.h5')
score = model.evaluate(
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

cm = confusion_matrix(test_generator.classes, np.argmax(predict,axis = -1))
print("Confusion Matrix:\n",cm)

plt.show()



