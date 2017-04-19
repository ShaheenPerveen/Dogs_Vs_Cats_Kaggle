## importing required lib
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
from keras.constraints import maxnorm
from keras.regularizers import l2
from random import shuffle

import cv2
import glob
import os
import re


train_path = "../Shaheen/trainDC"

ROWS = 128
COLS = 128
CHANNELS = 3

images = [img for img in os.listdir(train_path)]
images_dog = [img for img in os.listdir(train_path) if "dog" in img]
images_cat = [img for img in os.listdir(train_path) if "cat" in img]


train_list = images_dog + images_cat

shuffle(train_list)

train = np.ndarray(shape=(len(train_list),ROWS, COLS))
labels = np.ndarray(len(train_list))

for i, img_path in enumerate(train_list):
    img = cv2.imread(os.path.join(train_path, img_path), 0)
    img = cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    
    train[i] = img
    if "dog" in img_path:
        labels[i] = 1
    else:
        labels[i] = 0

## changing the shape of data
train1 = np.array(train).reshape((-1, 1, 128, 128)).astype('float32')

train1.shape

## dividing by 255
train1 /= 255


## splitting the data into train and test
x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size=0.3)

img_size = 128

## CNN

def cnn_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(1, img_size, img_size), activation='relu'))
    model.add(Convolution2D(32, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


model1 = cnn_model()


model1.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



## fitting the model
batch_size = 32
nb_epoch = 20

model1.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(x_test, y_test))


#validation

validation = model1.evaluate(x_test, y_test, verbose=1)
print('Test accuracy:', validation[1])


# model 2
img_size = 128

def catdog():
    
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(1, ROWS, COLS), activation='relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))


    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


model3 = catdog()


model3.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



## fitting the model
batch_size = 32
nb_epoch = 14

model3.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(x_test, y_test))


validation = model3.evaluate(x_test, y_test, verbose=1)
print('Test accuracy:', validation[1])


## data augmentation
X_train, X_val, Y_train, Y_val = train_test_split(train1, labels, test_size=0.2)


datagen = ImageDataGenerator(featurewise_center=False, 
                            featurewise_std_normalization=False, 
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10.,)

datagen1 = ImageDataGenerator(rotation_range=40,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True)


datagen1.fit(X_train)



# reinitialise the model

model2 = cnn_model()


model2.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

nb_epoch = 15
batch_size = 32
model2.fit_generator(datagen1.flow(X_train, Y_train, batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_val, Y_val))



validation = model2.evaluate(x_test, y_test, verbose=1)
print('Test accuracy:', validation[1])



## uploading and processing test data
test_path = "../Shaheen/testDC"

ROWS = 128
COLS = 128
CHANNELS = 3

images = [img for img in os.listdir(test_path)]

files = [ os.path.join('../Shaheen/testDC',str(i)+'.jpg') for i in range(1,12501) ]


test = np.ndarray(shape=(len(files),ROWS, COLS))

for i, img_path in enumerate(images):
    img = cv2.imread(os.path.join(test_path, img_path), 0)
    img = cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    
    test[i] = img
    


test1 = np.array(test).reshape((-1, 1, 128, 128)).astype('float32')


## dividing by 255
test1 /= 255

test1.shape



## predicting the test data
y_pred = model3.predict(test1)

# save results
np.savetxt('submission_DogsvCatsKaggleProb.csv', np.c_[range(1,len(test1)+1),y_pred], delimiter=',', header = 'id,label', comments = '')





