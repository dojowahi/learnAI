from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df


if __name__ == "__main__":
    # load the data from the file
    data = load_data("log_marks.csv", None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X = np.c_[np.ones((X.shape[0], 1)), X]
y = y[:, np.newaxis]
m,n= y.shape
model = LogisticRegression()
model.fit(X, y)
predicted_classes = model.predict(X)
y = y.reshape(m,)

accuracy = accuracy_score(y.flatten(), predicted_classes)
parameters = model.coef_

print(parameters, accuracy)