# imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# generate random data-set
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.rand(100, 1)


# plot
plt.scatter(X,y,s=10)
plt.xlabel('x')
plt.ylabel('y')
#plt.show()

ones = np.ones([X.shape[0], 1])
X = np.concatenate([ones, X],1)


# Setting Hyper parameters
alpha = 0.01
iters = 2000

# theta is a row vector
theta = np.zeros([2,1])


def computeCost(X, y, theta):
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp, 2)) / (2*len(X))


print("Cost function with default parameters:", computeCost(X, y, theta))

cost_history = []
iter_history = []

def gradientDescent(X, y, theta, alpha, iters):
    for _ in range(iters):
        temp = np.dot(X, theta) - y
        temp = np.dot(X.T, temp)
        theta = theta - (alpha/len(X)) * temp
        cost = computeCost(X, y, theta)
        cost_history.append(cost)
        iter_history.append(_)

    return theta, cost


theta,cost = gradientDescent(X, y, theta, alpha, iters)



X=X[:,1:]
plt.subplot(2,1,1)
plt.scatter(X, y)

y_vals = theta[0] + theta[1]* X
plt.plot(X, y_vals, '--')

plt.subplot(2,1,2)
plt.plot(iter_history,cost_history)

# mean squared error
mse = np.sum((y_vals - y)**2)

# root mean squared error

rmse = np.sqrt(mse/len(X))

# sum of square of residuals
ssr = np.sum((y_vals- y)**2)

#  total sum of squares
sst = np.sum((y - np.mean(y))**2)

# R2 score
r2_score = 1 - (ssr/sst)

print("Cost function after Gradient descent:", cost)
print("Theta0 or Intercept or Bias is:", theta[0])
print("Theta1 is:", theta[1])
print('Root mean squared error: ', rmse)
print('R2 score: ', r2_score)
plt.show()