from csv import reader, writer
import cv2
import numpy as np
from keras.models import Model, load_model

reader = open(r'filelist.txt', 'r')
writer = open('result.txt', 'w')

line = reader.readline()
path = 'test images/test images/'
model = load_model('menuset.h5')

mapping = {('burger', 0), ('dimsum', 1), ('ramen', 2), ('sushi', 3)}
mapping = dict((v, k) for k, v in mapping)
TARGET_SIZE = 128
while line:
    file_name = line.replace("\n", "")
    file_path = path + file_name
    print(file_path)

    im = cv2.imread(file_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_resized = cv2.resize(im, (TARGET_SIZE, TARGET_SIZE))
    im = im_resized/255.
    # im_reshape = im.reshape(1, TARGET_SIZE, TARGET_SIZE, 3)

    predict = model.predict(im)
    print(predict)
    predict_class_idx = np.argmax(predict, axis=-1)[0]
    predict_class_name = mapping[predict_class_idx]

    text = file_name + "::" + predict_class_name[0].upper() + '\n'
    print(text)
    writer.writelines(text)

    line = reader.readline()