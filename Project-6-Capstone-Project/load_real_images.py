
# import numpy as np, h5py 
# f = h5py.File('./capstone/test_32x32.mat','r') 
# data = f.get('data/variable1') 
# data = np.array(data) # For converting to numpy array
# print data
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import mnist_list as mnist
from itertools import islice
    
from pylab import *
from numpy import *

train = sio.loadmat('./capstone/train_32x32.mat')
test = sio.loadmat('./capstone/test_32x32.mat')

X_train = train.get('X') 
y_train = train.get('y') 

X_test = test.get('X') 
y_test = test.get('y') 


images_train = X_train
# images_train = images_train/255.0
labels_train = y_train

images_test = X_test
# images_test = images_test/255.0
labels_test = y_test


#plotting several images on the same figure:
nrows, ncols = 4, 4

fig = plt.figure()    
for i in range(1,17):
    ax = fig.add_subplot(nrows, ncols, i)
    ax.imshow(images_train[:,:,:,i])
    plt.xticks(())
    plt.yticks(())

#uncomment the line below to show the grid with images of digits:    
# plt.show()


