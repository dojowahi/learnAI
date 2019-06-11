import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#Load Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

#Check the shape of data
print (X_iris.shape)
print (y_iris.shape)

plt.scatter(X_iris[:,0], X_iris[:,3],c = y_iris, cmap='Dark2')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size = 0.20, random_state = 82)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print(y_pred)

y_compare = np.vstack((y_test,y_pred)).T
cm = confusion_matrix(y_test, y_pred)
print(cm)