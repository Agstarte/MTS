import numpy as np

from sklearn.feature_selection import r_regression


dataset = np.genfromtxt("wisconsin.csv", delimiter=",")
X = dataset[1:, :-1]
y = dataset[1:, -1].astype(int)

print(r_regression(X, y, center=True))
