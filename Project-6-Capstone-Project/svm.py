import vect_data 

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer

# clf = svm.SVC()
parameters = {'kernel': ('linear', 'rbf'), 'C': [1,10,100, 1000], 'gamma': [0.001, 0.0001]}

svr = svm.SVC()
clf = GridSearchCV(svr, parameters)

train_dataset = vect_data.images_train
nsamples, nx, ny = train_dataset.shape
d2_train_dataset = train_dataset.reshape((nsamples,nx*ny))

y_train = vect_data.labels_train.ravel()

clf.fit(d2_train_dataset, y_train)


test_dataset = vect_data.images_test
t_nsamples, t_nx, t_ny = test_dataset.shape
d2_test_dataset = test_dataset.reshape((t_nsamples,t_nx*t_ny))

y_test = vect_data.labels_test.ravel()

result = clf.predict(d2_test_dataset)

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

