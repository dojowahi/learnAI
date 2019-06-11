import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt    # more on this later
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


data = pd.read_csv("ex2data2.txt", header = None)
X = data.iloc[:,:-1]
y = data.iloc[:,2]

good = data.loc[y == 1]
bad = data.loc[y == 0]
plt.scatter(good.iloc[:, 0], good.iloc[:, 1], s=10, label='Good')
plt.scatter(bad.iloc[:, 0], bad.iloc[:, 1], s=10, label='Bad')
plt.legend()
#plt.show()


def mapFeature(X1, X2):
    degree = 6
    out = np.ones(X.shape[0])[:,np.newaxis]
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j),np.power(X2, j))[:,np.newaxis]))
    return out


X = mapFeature(X.iloc[:,0], X.iloc[:,1])

(m, n) = X.shape
theta = np.zeros((n,1))
y = y[:, np.newaxis]
lambda_r = 1

# With Scikit

model = LogisticRegression()
model.fit(X, y)
predicted_classes = model.predict(X)
y = y.reshape(m,)

accuracy = accuracy_score(y.flatten(), predicted_classes)
parameters = model.coef_

print("With Scikit - accuracy:", accuracy)


def sigmoid(x):
  return 1/(1+np.exp(-x))


def regCostFunction(theta, X, y, lambda_r):
    J = (-1/m) * np.sum(np.multiply(y, np.log(sigmoid(np.dot(X, theta))))+ np.multiply((1-y), np.log(1 - sigmoid(np.dot(X, theta)))))
    reg = (lambda_r / (2 * m)) * np.dot(theta[1:].T, theta[1:])
    return J+reg


def regGradient(theta, X, y, lambda_r):
    m=len(y)
    grad = (1/m) * X.T @ (sigmoid(X @ theta) - y)
    grad[1:] = grad[1:] + (lambda_r / m) * theta[1:]
    return grad


J = regCostFunction(theta, X, y, lambda_r)
print(J)

res = minimize(regCostFunction, theta, args=(X, y, 0), method=None, jac=regGradient, options={'maxiter':3000})

#theta_opt = opt_weights[0]

def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))

accuracy = 100*sum(predict(res.x, X) == y.ravel())/y.size

print(accuracy)

#pred = [sigmoid(np.dot(X, theta)) >= 0.5]
#print(np.mean(pred == y.flatten()) * 100)

