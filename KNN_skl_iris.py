from __future__ import print_function
import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
np.random.seed(7)
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=130)

def myweight(distances):
    sigma2 = .4
    return np.exp(-distances**2/sigma2)


model = neighbors.KNeighborsClassifier(n_neighbors=9, p=2, weights=myweight)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

for i in range(len(y_pred)):
    print(y_pred[i], ' ', y_test[i])

print(100*accuracy_score(y_pred, y_test))