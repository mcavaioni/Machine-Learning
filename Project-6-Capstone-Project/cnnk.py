import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import scipy.io as sio
from itertools import islice
from pylab import *
from numpy import *
import matplotlib.image as mpimg
from sklearn.cross_validation import train_test_split
from sklearn import datasets

train = sio.loadmat('./capstone/train_32x32.mat')
test = sio.loadmat('./capstone/test_32x32.mat')

X_train = train.get('X') 
y_train = train.get('y') 

X_test = test.get('X') 
y_test = test.get('y') 



#############
#Model creation:

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

seed = 7
np.random.seed(seed)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0


nx, ny, nz, nsamples = X_train.shape
nx_t, ny_t, nz_t, nsamples_t = X_test.shape

X_train = X_train.reshape((nsamples,nz, nx, ny))
X_test = X_test.reshape((nsamples_t, nz_t, nx_t, ny_t))

X_train_fin, X_val, y_train_fin, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

y_train_fin = np_utils.to_categorical(y_train_fin)
y_val = np_utils.to_categorical(y_val)
num_classes = y_val.shape[1]



from keras.layers import Dense, Activation
from keras.optimizers import SGD

model = Sequential()

#This was a simple model firstly implemented:
# model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(num_classes, activation='softmax'))
# model.add(Activation('relu'))


model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1, momentum=0.9, nesterov=True), metrics=['accuracy'])
print(model.summary())

model.fit(X_train_fin, y_train_fin, nb_epoch=20, batch_size=32, validation_data=(X_val, y_val))
loss_and_metrics = model.evaluate(X_val, y_val, batch_size=32)
print("Accuracy: %.2f%%" % (loss_and_metrics[1]*100))

from keras.models import load_model
from keras.models import model_from_json
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

y_test = np_utils.to_categorical(y_test)

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, batch_size=32, verbose=1)
print("Test accuracy: %.2f%%" % (score[1]*100))

