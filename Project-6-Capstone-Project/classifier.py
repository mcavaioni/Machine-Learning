import vect_data 
from sklearn.datasets import load_digits

from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt


clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(200,200), random_state=1, max_iter=2000, activation='logistic', learning_rate_init=0.1)

#In case of single hidden layer (with 100 neurons) the classifier used is the following:
# clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(100,), random_state=1, max_iter=2000, activation='logistic', learning_rate_init=0.1)

#reshaping the training and testing datasets, to fit to the classifier:
train_dataset = vect_data.images_train
nsamples, nx, ny = train_dataset.shape
d2_train_dataset = train_dataset.reshape((nsamples,nx*ny))

y_train = vect_data.labels_train.ravel()

#fit the input data to the classifier:
clf.fit(d2_train_dataset, y_train)


test_dataset = vect_data.images_test
t_nsamples, t_nx, t_ny = test_dataset.shape
d2_test_dataset = test_dataset.reshape((t_nsamples,t_nx*t_ny))

y_test = vect_data.labels_test.ravel()

#predict results on the test set:
result = clf.predict(d2_test_dataset)

#count how many correct labels are predicted, evaluating the prediction against the test label set:
count=[]
for i in range(0,len(d2_test_dataset)):
  if result[i]==vect_data.labels_test[i]:
    count.append(1)
  else:
    count.append(0) 

print count.count(1)

#accuracy on training set
acc_training = clf.score(d2_train_dataset, y_train)
print("Training set score: %f" % acc_training)

#accuracy on test set
acc_test = clf.score(d2_test_dataset, y_test)
print("Test set score: %f" % acc_test)

#to show first layer weights uncomment lines below:

# fig, axes = plt.subplots(4, 4)
# # use global min / max to ensure all weights are shown on the same scale
# vmin, vmax = clf.coefs_[0].min(), clf.coefs_[0].max()
# for coef, ax in zip(clf.coefs_[0].T, axes.ravel()):
#     ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
#                vmax=.5 * vmax)
#     ax.set_xticks(())
#     ax.set_yticks(())
# plt.show()

