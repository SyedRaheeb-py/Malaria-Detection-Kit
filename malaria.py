# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 23:24:52 2022

@author: SyedRaheeb
"""
import numpy as np
from numpy import random, argmax, array, arange
from tensorflow.python.keras.utils import np_utils
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import img_to_array
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import Sequential
from keras import backend as K
from sklearn.utils import class_weight

print(os.listdir("Malaria-Dataset"))
parasitized_data = os.listdir("Malaria-Dataset/Parasitized")
print(parasitized_data[:10])
uninfected_data = os.listdir("Malaria-Dataset/Uninfected")
print('\n')
print(uninfected_data[:10])

print("Parasitized:", len(parasitized_data))
print("Uninfected:", len(uninfected_data))

if(os.path.isdir('output') == False):
    os.mkdir('output')
    if(os.path.isdir('output/Parasitized') == False):
        os.mkdir('output/Parasitized')
    if(os.path.isdir('output/Uninfected') == False):
        os.mkdir('output/Uninfected')

plt.figure(figsize = (12,12))
for i in range(4):
    plt.subplot(1, 4, i+1)
    img = cv2.imread("Malaria-Dataset/Parasitized" + "/" + parasitized_data[i])
    plt.imshow(img)
    plt.title('PARASITIZED : 1')
    plt.tight_layout()
    plt.savefig('output/Parasitized/Parasitized.png')

plt.figure(figsize = (12,12))
for i in range(4):
    plt.subplot(1, 4, i+1)
    img = cv2.imread("Malaria-Dataset/Uninfected" + "/" + uninfected_data[i+1])
    plt.imshow(img)
    plt.title('UNINFECTED : 0')
    plt.tight_layout()
    plt.savefig('output/Uninfected/Uninfected.png')

data = []
labels = []
for img in parasitized_data:
    try:
        img_read = plt.imread("Malaria-Dataset/Parasitized" + "/" + img)
        img_resize = cv2.resize(img_read, (50, 50))
        img_array = img_to_array(img_resize)
        print("TRAINING: ", img_array.dtype, img_array.min(), img_array.max())
        data.append(img_array)
        labels.append(1)
    except:
        None
        
for img in uninfected_data:
    try:
        img_read = plt.imread("Malaria-Dataset/Uninfected" + "/" + img)
        img_resize = cv2.resize(img_read, (50, 50))
        img_array = img_to_array(img_resize)
        print("TRAINING: ", img_array.dtype, img_array.min(), img_array.max())
        data.append(img_array)
        labels.append(0)
    except:
        None
#plt.imshow(data[0])
#plt.show()

image_data = array(data)
labels = array(labels)
idx = arange(image_data.shape[0])
random.shuffle(idx)
image_data = image_data[idx]
labels = labels[idx]

x_train, x_test, y_train, y_test = train_test_split(image_data, labels, test_size = 0.25, random_state = 101, stratify=labels)
y_train = np_utils.to_categorical(y_train, num_classes = 2)
y_test = np_utils.to_categorical(y_test, num_classes = 2)
print(f'SHAPE OF TRAINING IMAGE DATA : {x_train.shape}')
print(f'SHAPE OF TESTING IMAGE DATA : {x_test.shape}')
print(f'SHAPE OF TRAINING LABELS : {y_train.shape}')
print(f'SHAPE OF TESTING LABELS : {y_test.shape}')

def CNNbuild(height, width, classes, channels):
    model = Sequential()
    
    inputShape = (height, width, channels)
    chanDim = -1
    
    if K.image_data_format() == 'channels_first':
        inputShape = (channels, height, width)
    model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = inputShape))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization(axis = chanDim))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization(axis = chanDim))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization(axis = chanDim))
    model.add(Dropout(0.2))

    model.add(Flatten())
    
    model.add(Dense(512, activation = 'relu'))
    model.add(BatchNormalization(axis = chanDim))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation = 'softmax'))
    
    return model

height = 50
width = 50
classes = 2
channels = 3
model = CNNbuild(height = height, width = width, classes = classes, channels = channels)
model.summary()


model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
y_labels = np.argmax(y_train, axis=1)
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_labels), y=y_labels)
h = model.fit(x_train, y_train, epochs=8, batch_size=16, class_weight={0: weights[0], 1: weights[1]})

plt.figure(figsize = (18,8))
plt.plot(range(8), h.history['accuracy'], label = 'Training Accuracy')
plt.plot(range(8), h.history['loss'], label = 'Training Loss')
plt.xlabel("Number of Epoch's")
plt.ylabel('Accuracy/Loss Value')
plt.title('Training Accuracy and Training Loss')
plt.legend(loc = "best")
plt.savefig('output/Training Accuracy and Training Loss.png')
if(plt.savefig('output/Training Accuracy and Training Loss.png')):
    print('Training Accuracy and Training Loss Images Saved Successfully')

predictions = model.evaluate(x_test, y_test)
print(f'LOSS : {predictions[0]}')
print(f'ACCURACY : {predictions[1]}')
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

model.save("models/model.h5")


