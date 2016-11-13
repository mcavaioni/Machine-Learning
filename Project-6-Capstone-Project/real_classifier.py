import load_real_images
import convolution as conv
import numpy as np
from sklearn.datasets import load_digits

from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt

import edges

clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(100,), random_state=1, max_iter=2000, activation='logistic', learning_rate_init=0.1)


train_dataset = np.array([conv.gray_list])

nx, nsamples, ny, nz = train_dataset.shape
d2_train_dataset = train_dataset.reshape((nsamples,nx*ny*nz))

#creating the labels in vectorized form:
y_train_vect = load_real_images.labels_train.ravel()

clf.fit(d2_train_dataset, y_train_vect)


test_dataset = np.array([conv.gray_test_list])
nx_t, nsamples_t, ny_t, nz_t = test_dataset.shape
d2_test_dataset = test_dataset.reshape((nsamples_t,nx_t*ny_t*nz_t))

#creating the labels in vectorized form:
y_test = load_real_images.labels_test.ravel()

result = clf.predict(d2_test_dataset)

count=[]
for i in range(0,len(d2_test_dataset)):
  if result[i]==load_real_images.labels_test[i]:
    count.append(1)
  else:
    count.append(0) 

#counting correct match
print count.count(1)

#accuracy on training set
acc_training = clf.score(d2_train_dataset, y_train_vect)
print("Training set score: %f" % acc_training)

#accuracy on test set
acc_test = clf.score(d2_test_dataset, y_test)
print("Test set score: %f" % acc_test)



