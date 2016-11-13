import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2
import load_real_images as lri

gray_list=[]
for i in range(0,73257):
  train_dataset = lri.images_train[:,:,:,i]

  # construct a sharpening filter
  sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

  # construct the Laplacian kernel used to detect edge-like
  # regions of an image
  laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")
   
  # construct the Sobel x-axis kernel
  sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")
   
  # construct the Sobel y-axis kernel
  sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")

  # construct the kernel bank
  kernelBank = laplacian

  from skimage.color import rgb2gray

  image = train_dataset*255
  gray = rgb2gray(image)

  opencvOutput = cv2.filter2D(gray, -1, kernelBank) 

  #create the list of all images transformed in greyscale:
  gray_list.append(opencvOutput)
  
#same process as above, for the test set:
gray_test_list=[]
for i in range(0,26032):
  test_dataset = lri.images_test[:,:,:,i]

  # construct a sharpening filter
  sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

  # construct the Laplacian kernel used to detect edge-like
  # regions of an image
  laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")
   
  # construct the Sobel x-axis kernel
  sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")
   
  # construct the Sobel y-axis kernel
  sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")

  # construct the kernel bank:
  kernelBank = laplacian

  from skimage.color import rgb2gray

  image_test = test_dataset*255
  gray_test = rgb2gray(image_test)

  opencvOutput_test = cv2.filter2D(gray_test, -1, kernelBank) 

  #create the list of all images transformed in greyscale:
  gray_test_list.append(opencvOutput_test)

