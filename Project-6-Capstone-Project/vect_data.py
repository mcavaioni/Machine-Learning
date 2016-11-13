import mnist_list as mnist
from itertools import islice
    
from pylab import *
from numpy import *
import matplotlib.pyplot as plt

images_train, labels_train = mnist.load_mnist('training')
images_train = images_train/255.0

images_test, labels_test = mnist.load_mnist('testing')
images_test = images_test/255.0


#plotting several images on the same figure:
nrows, ncols = 4, 4

fig = plt.figure()    
for i in range(1,17):
    ax = fig.add_subplot(nrows, ncols, i)
    ax.imshow(images_train[i])
    plt.xticks(())
    plt.yticks(())

#uncomment the line below to show the grid with images of digits:    
# plt.show()



