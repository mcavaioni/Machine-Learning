
#Image SVHN:
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import scipy.io 
from itertools import islice
from pylab import *
from numpy import *
import matplotlib.image as mpimg
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.utils import np_utils



X_train = scipy.io.loadmat('./capstone/train_32x32.mat')['X']
y_train = scipy.io.loadmat('./capstone/train_32x32.mat')['y']
X_test = scipy.io.loadmat('./capstone/test_32x32.mat')['X']
y_test = scipy.io.loadmat('./capstone/test_32x32.mat')['y']


#normalizing:

# X_train = X_train.astype('float32') / 128.0 - 1
# X_test = X_test.astype('float32') / 128.0 - 1


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
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#############################################

#reshape images to format (n_samples, 32,32,3) and creating onw-hot encoding for labels
def reformat(data, Y):
    xtrain = []
    trainLen = data.shape[3]
    for x in range(trainLen):
        xtrain.append(data[:,:,:,x])
    xtrain = np.asarray(xtrain)
    Ytr=[]
    for el in Y:
        temp=np.zeros(10)
        if el==10:
            temp[0]=1
        elif el==1:
            temp[1]=1
        elif el==2:
            temp[2]=1
        elif el==3:
            temp[3]=1
        elif el==4:
            temp[4]=1
        elif el==5:
            temp[5]=1
        elif el==6:
            temp[6]=1
        elif el==7:
            temp[7]=1
        elif el==8:
            temp[8]=1
        elif el==9:
            temp[9]=1
        Ytr.append(temp)
    return xtrain, np.asarray(Ytr)

X_train, y_train = reformat(X_train, y_train)
X_test, y_test = reformat(X_test, y_test)

###########################################
#transform in gray images:
# import cv2
# def grayscale(image):
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return image_gray

# X_train_gray=[]
# for i in range(len(X_train)):
#     X_train_gray.append(grayscale(X_train[i]))
# X_train_gray = np.array(X_train_gray)
# X_train = X_train_gray[:,:,:,np.newaxis]

# X_test_gray=[]
# for i in range(len(X_test)):
#     X_test_gray.append(grayscale(X_test[i]))
# X_test_gray = np.array(X_test_gray)
# X_test = X_test_gray[:,:,:,np.newaxis]
##############################################


#split train/val/test:
from sklearn.cross_validation import train_test_split
X_train_fin, X_val, y_train_fin, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=42)

#######################################
import tensorflow as tf

# Parameters
learning_rate = 0.001
batch_size = 16
training_epochs = 10000

n_input = 3072  #  data input (Shape: 32*32) - 3072 if 3 channels (otherwise 1024)
n_classes = 10  #  total classes (0-9 digits)  10 for SVHN

layer_width = {
    'layer_1': 16,
    'layer_2': 16,
    # 'layer_3': 128,
    'fully_connected': 128
}

# Store layers weight & bias
weights = {
    'layer_1': tf.Variable(tf.truncated_normal(
        [5, 5, 3, layer_width['layer_1']], stddev=0.1)),    # change to [5,5,3,.... if 3 channels
    'layer_2': tf.Variable(tf.truncated_normal(
        [5, 5, layer_width['layer_1'], layer_width['layer_2']], stddev=0.1)),
    # 'layer_3': tf.Variable(tf.truncated_normal(
    #     [5, 5, layer_width['layer_2'], layer_width['layer_3']])),
    'fully_connected': tf.Variable(tf.truncated_normal(
        [1024, layer_width['fully_connected']], stddev=0.1)),       
    'out': tf.Variable(tf.truncated_normal(
        [layer_width['fully_connected'], n_classes], stddev=0.1))
}
biases = {
    'layer_1': tf.Variable(tf.constant(1.0, shape=[layer_width['layer_1']])),
    'layer_2': tf.Variable(tf.constant(1.0, shape=[layer_width['layer_2']])),
    # 'layer_3': tf.Variable(tf.zeros(layer_width['layer_3'])),
    'fully_connected': tf.Variable(tf.constant(1.0, shape=[layer_width['fully_connected']])),
    'out': tf.Variable(tf.constant(1.0, shape=[n_classes]))
}


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


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
    # conv3 = conv2d(conv2, weights['layer_3'], biases['layer_3'])
    # conv3 = maxpool2d(conv2)

    # Fully connected layer
    # Reshape conv3 output to fit fully connected layer input
    shape = conv2.get_shape().as_list()
    fc1 = tf.reshape(conv2, [-1, shape[1] * shape[2] * shape[3]])
    # fc1 = tf.reshape(
    #     conv2,
    #     [-1, weights['fully_connected'].get_shape().as_list()[0]])
    fc1 = tf.add(
        tf.matmul(fc1, weights['fully_connected']),
        biases['fully_connected'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, 0.93)

    # Output Layer - class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# tf Graph input
x = tf.placeholder("float", [None, 32, 32, 3])   #change to ...[None, 32, 32, 1]) if 1 channel
y = tf.placeholder("float", [None, n_classes])

dropout = tf.placeholder(tf.float32)

logits = conv_net(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    

train_prediction = tf.nn.softmax(logits)

saver = tf.train.Saver()
####################

#evaluate accuracy on the trianing set
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


with tf.Session() as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    average = 0
    for step in range(training_epochs):
        #   Constucting the batch from the data set
        offset = (step * batch_size) % (y_train_fin.shape[0] - batch_size)
        batch_data = X_train_fin[offset:(offset + batch_size), :, :, :]
        batch_labels = y_train_fin[offset:(offset + batch_size), :]
        #   Dictionary to be fed to TensorFlow Session
        feed_dict = {x : batch_data, y : batch_labels, dropout: 0.93}
        _, l, predictions = session.run([optimizer, cost, train_prediction], feed_dict=feed_dict)
        #   Calculating the Accuracy of the predictions
        accu = accuracy(predictions, batch_labels)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accu)
        average += accu
    print("Average Training Accuracy : ", (average / training_epochs))


    # evaluate accuracy on the validation set:
    average = 0
    for step in range(training_epochs):
        #   Constucting the batch from the data set
        offset = (step * batch_size) % (y_val.shape[0] - batch_size)
        batch_data = X_val[offset:(offset + batch_size), :, :, :]
        batch_labels = y_val[offset:(offset + batch_size), :]
        #   Dictionary to be fed to TensorFlow Session
        feed_dict = {x : batch_data, y : batch_labels, dropout: 0.93}
        _, l, predictions = session.run([optimizer, cost, train_prediction], feed_dict=feed_dict)
        #   Calculating the Accuracy of the predictions
        accu = accuracy(predictions, batch_labels)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accu)
        average += accu
    print("Average Validation Accuracy : ", (average / training_epochs))

    #saving the model:
    save_path = saver.save(session, "/tmp/SVHNmodel.ckpt")
    print("model saved in file: %s" % save_path)

##############

# evaluating the accuracy on the test set using the saved model:
with tf.Session() as session:
    tf.initialize_all_variables().run()
    saver.restore(session, "/tmp/SVHNmodel.ckpt")
    average = 0
    for step in range(training_epochs):
        #   Constucting the batch from the data set
        offset = (step * batch_size) % (y_test.shape[0] - batch_size)
        batch_data = X_test[offset:(offset + batch_size), :, :, :]
        batch_labels = y_test[offset:(offset + batch_size), :]
        #   Dictionary to be fed to TensorFlow Session
        logits = conv_net(x, weights, biases)
        train_prediction = tf.nn.softmax(logits)
        feed_dict = {x : batch_data, y : batch_labels, dropout: 0.93}
        predictions = session.run(train_prediction, feed_dict=feed_dict)
        #   Calculating the Accuracy of the predictions
        accu = accuracy(predictions, batch_labels)
        if (step % 50 == 0):
            # print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accu)
        average += accu
    print("Average Testing Accuracy : ", (average / training_epochs))

#printing the probabilities of predictions of each label for the first 10 test images:
with tf.Session() as session:
    tf.initialize_all_variables().run()
    saver.restore(session, "/tmp/SVHNmodel.ckpt")
    p = train_prediction.eval(feed_dict={x: X_test[0:10]})
    print("predictions", p)

#printing the values in order of predictions (for the first 10 test images)
with tf.Session() as session:
    tf.initialize_all_variables().run()
    saver.restore(session, "/tmp/SVHNmodel.ckpt")
    logits = conv_net(X_test[0:10], weights, biases)
    values, indices = session.run(tf.nn.top_k(logits,10))
    print(values)
    print(indices)

print(y_test[0:10]) #5 2 1 0 6 1  9 1 1 8
