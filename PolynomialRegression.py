import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(0)
X = 2 - 3 * np.random.normal(0, 1, 20)
y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-3, 3, 20)
print(X.shape)

X = X.reshape(-1,1)
y = y.reshape(-1,1)


model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.subplot(2,1,1)
plt.scatter(X, y, s=10)
plt.plot(X, y_pred, color='r')
plt.title('Linear Regression')

# model evaluation
rmse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print('Root mean squared error: ', rmse)
print('R2 score: ', r2)

polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(X)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
print(rmse)
print(r2)

plt.subplot(2,1,2)
plt.scatter(X, y, s=10)

# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X,y_poly_pred), key=sort_axis)
X, y_poly_pred = zip(*sorted_zip)
plt.plot(X, y_poly_pred, color='m')
plt.show()
