import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt

def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df


if __name__ == "__main__":
    # load the data from the file
    data = load_data("log_marks.csv", None)

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

admitted = data.loc[y == 1]
not_admitted = data.loc[y == 0]

plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
plt.legend()
#plt.show()

m = len(X)
(m, n) = X.shape

# Adding 1 as first column to handle bias
X = np.hstack((np.ones((m,1)), X))
y = y[:, np.newaxis]

# theta is a row vector
theta = np.zeros((n+1,1))

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))


def costFunction(theta, X, y):
    J = (-1/m) * np.sum(np.multiply(y, np.log(sigmoid(np.dot(X, theta))))+ np.multiply((1-y), np.log(1 - sigmoid(np.dot(X, theta)))))
    return J


def gradient(theta, X, y):
    return (1/m) * np.dot(X.T, (sigmoid(np.dot(X, theta)) - y))


J = costFunction(theta, X, y)
print("Initial Cost", J)


opt_weights = opt.fmin_tnc(func=costFunction, x0=theta, fprime=gradient,args=(X, y.flatten()))
theta_opt = opt_weights[0]
J = costFunction(theta_opt[:, np.newaxis], X, y)
print("Final Cost:", J)


# Prediction
def accuracy(X, y, theta,threshold):
    pred_class = [sigmoid(np.dot(X, theta))>= threshold]
    acc = np.mean(pred_class == y)
    print(acc * 100)


accuracy(X, y.flatten(), theta_opt,0.5)

# Decision line
# x = np.linspace(-6, 60, 50)
x = [np.min(X[:, 1] - 5), np.max(X[:, 2] + 5)]

marks2 = - (theta_opt[0] + np.dot(theta_opt[1], X)) / theta_opt[2]
plt.plot(X, marks2, label='Decision Boundary')
plt.show()


