import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

##Data load
mnist_data = np.load('mnist.npz')
x_test = mnist_data['x_test']
x_train = mnist_data['x_train']
y_test = mnist_data['y_test']
y_train = mnist_data['y_train']
Xtest = np.reshape(x_test,(len(x_test),-1))
Xtrain = np.reshape(x_train,(len(x_train),-1))
Ytest = np.reshape(y_test,(len(y_test),-1))
Ytrain = np.reshape(y_train,(len(y_train),-1))

plt.title('Handwritten Digits')
plt.imshow(x_test[0])
plt.show()
clf = DecisionTreeClassifier(criterion='gini')
clf.fit(Xtrain,Ytrain)
pre_result = clf.predict(Xtest)
score = accuracy_score(pre_result,Ytest)
print("准确率",score)