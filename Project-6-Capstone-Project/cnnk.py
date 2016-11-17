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

X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
X_train = X_train / 255.0
X_test = X_test / 255.0


nx, ny, nz, nsamples = X_train.shape
nx_t, ny_t, nz_t, nsamples_t = X_test.shape

# X_train = X_train.reshape((nsamples,nz, nx, ny))
# X_test = X_test.reshape((nsamples_t, nz_t, nx_t, ny_t))
X_train = np.transpose(X_train,(3,2,0,1))
X_test = np.transpose(X_test,(3,2,0,1))


X_train_fin, X_val, y_train_fin, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

y_train_fin = np_utils.to_categorical(y_train_fin)
y_val = np_utils.to_categorical(y_val)
num_classes = y_val.shape[1]


from keras.layers import Dense, Activation
from keras.optimizers import SGD


#This was a simple model firstly implemented:
# model = Sequential()
# model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(num_classes, activation='softmax'))
# model.add(Activation('relu'))

# LeNet network
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


#complex network
# model = Sequential()
# model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), activation='relu', border_mode='same'))
# model.add(Dropout(0.2))
# model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
# model.add(Dropout(0.2))
# model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
# model.add(Dropout(0.2))
# model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dropout(0.2))
# model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(Dense(num_classes, activation='softmax'))

#compile the model using SGD as optimizer
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1, momentum=0.9, nesterov=True), metrics=['accuracy'])
print(model.summary())


#fit the model to the training data and use the validation set as validation data:
model.fit(X_train_fin, y_train_fin, nb_epoch=1, batch_size=32, validation_data=(X_val, y_val))
loss_and_metrics = model.evaluate(X_val, y_val, batch_size=32)
print("Accuracy: %.2f%%" % (loss_and_metrics[1]*100))

# These lines of code save to disk the fitted model, so there is no need to re-fit it for future predictions:
from keras.models import load_model
from keras.models import model_from_json
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# These lines of code re-load the model previously saved:
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
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, batch_size=32, verbose=1)
print("Test accuracy: %.2f%%" % (score[1]*100))

####

# This loops allows to display a set of 10 random images, dispalying the image and the prediction.
# In addition, it prints out the probability associated to each digit.

for i in np.random.choice(np.arange(0, len(y_test)), size=(10,)):
  # re-shape the inputs so to fit into the prediction model
  X_test = X_test.reshape((nsamples_t, nz_t, nx_t, ny_t))
  
  # classify the digit, assigning a probability to each one of them
  probs = loaded_model.predict(X_test[i, np.newaxis])
  # it selects the digit that has highest probability
  prediction = probs.argmax(axis=1)
  print(probs)
  #re-shape the X_test so to recompile the picture  
  X_test = X_test.reshape((nx_t, ny_t, nz_t, nsamples_t))
  
  image = (X_test[:,:,:,i])
  # resize the image from a 32 x 32 image to a 96 x 96 image so we can better see it
  image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR) 
  cv2.putText(image, str(prediction[0]), (5, 20),
    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
 
  # show the image and prediction
  print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
    np.argmax(y_test[i])))
  cv2.imshow("Digit", image)
  cv2.waitKey(0)