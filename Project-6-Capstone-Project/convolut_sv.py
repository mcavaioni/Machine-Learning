# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

# print(mnist.train.labels.shape)
################################

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
from keras.utils import np_utils


train = sio.loadmat('./capstone/train_32x32.mat')
test = sio.loadmat('./capstone/test_32x32.mat')

X_train = train.get('X') 
y_train = train.get('y') 

X_test = test.get('X') 
y_test = test.get('y') 

nx, ny, nz, nsamples = X_train.shape
X_train = X_train.reshape((nsamples, nx, ny, nz))

nx_t, ny_t, nz_t, nsamples_t = X_test.shape
X_test = X_test.reshape((nsamples_t, nx_t, ny_t, nz_t))

for n,i in enumerate(y_test):
  if i==10:
    y_test[n]=0

for n,i in enumerate(y_train):
  if i==10:
    y_train[n]=0

y_test = np_utils.to_categorical(y_test)
y_train = np_utils.to_categorical(y_train)


#transform in gray images:
import cv2
def grayscale(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image_gray

X_train_gray=[]
for i in range(len(X_train)):
    X_train_gray.append(grayscale(X_train[i]))
X_train_gray = np.array(X_train_gray)
X_train = X_train_gray[:,:,:,np.newaxis]

X_test_gray=[]
for i in range(len(X_test)):
    X_test_gray.append(grayscale(X_test[i]))
X_test_gray = np.array(X_test_gray)
X_test = X_test_gray[:,:,:,np.newaxis]
##############################################

#normalize:
def normalize(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    """
    a=0.1
    b=0.9
    x_min = 0
    x_max =255   
    return (a + (((image_data)*(b-a))/(x_max-x_min)))

X_train = normalize(X_train)
X_test = normalize(X_test)
#############################################

#split train/val/test:
from sklearn.cross_validation import train_test_split
X_train_fin, X_val, y_train_fin, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=42)

X_train = X_train_fin
y_train = y_train_fin
#############################################

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#######################################
import tensorflow as tf

# Parameters
learning_rate = 0.2
batch_size = 220
training_epochs = 8

n_input = 1024  #  data input (Shape: 32*32) - 3072 if 3 channels
n_classes = 10  #  total classes (0-9 digits)

layer_width = {
    'layer_1': 32,
    'layer_2': 64,
    'layer_3': 128,
    'fully_connected': 512
}

# Store layers weight & bias
weights = {
    'layer_1': tf.Variable(tf.truncated_normal(
        [5, 5, 1, layer_width['layer_1']])),    # change to [5,5,3,.... if 3 channels
    'layer_2': tf.Variable(tf.truncated_normal(
        [5, 5, layer_width['layer_1'], layer_width['layer_2']])),
    'layer_3': tf.Variable(tf.truncated_normal(
        [5, 5, layer_width['layer_2'], layer_width['layer_3']])),
    'fully_connected': tf.Variable(tf.truncated_normal(
        [1024, layer_width['fully_connected']])),
    'out': tf.Variable(tf.truncated_normal(
        [layer_width['fully_connected'], n_classes]))
}
biases = {
    'layer_1': tf.Variable(tf.zeros(layer_width['layer_1'])),
    'layer_2': tf.Variable(tf.zeros(layer_width['layer_2'])),
    'layer_3': tf.Variable(tf.zeros(layer_width['layer_3'])),
    'fully_connected': tf.Variable(tf.zeros(layer_width['fully_connected'])),
    'out': tf.Variable(tf.zeros(n_classes))
}


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.tanh(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')

# Create model
def conv_net(x, weights, biases):
    # Layer 1
    conv1 = conv2d(x, weights['layer_1'], biases['layer_1'])
    conv1 = maxpool2d(conv1)

    # Layer 2
    conv2 = conv2d(conv1, weights['layer_2'], biases['layer_2'])
    conv2 = maxpool2d(conv2)

    # Layer 3
    conv3 = conv2d(conv2, weights['layer_3'], biases['layer_3'])
    conv3 = maxpool2d(conv2)

    # Fully connected layer
    # Reshape conv3 output to fit fully connected layer input
    fc1 = tf.reshape(
        conv3,
        [-1, weights['fully_connected'].get_shape().as_list()[0]])
    fc1 = tf.add(
        tf.matmul(fc1, weights['fully_connected']),
        biases['fully_connected'])
    fc1 = tf.nn.tanh(fc1)

    # Output Layer - class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# tf Graph input
x = tf.placeholder("float", [None, 32, 32, 1])   #change to ...[None, 32, 32, 3]) if 3 channels
y = tf.placeholder("float", [None, n_classes])

logits = conv_net(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    .minimize(cost)


# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int((len(X_train))/batch_size)
        # Loop over all batches
        init_val = 0
        final_val = batch_size
        for i in range(total_batch):
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            if final_val<len(X_train):
                batch_x = X_train[init_val:final_val]
                batch_y = y_train[init_val:final_val]
                init_val = init_val + batch_size
                final_val = final_val + batch_size
            else:
                batch_x = X_train[init_val:len(X_train)]
                batch_y = y_train[init_val:len(X_train)]

            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        # Display logs per epoch step
        c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(
        "Validation Accuracy:",
        accuracy.eval({x: X_val, y: y_val}))