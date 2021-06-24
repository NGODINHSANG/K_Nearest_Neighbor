import numpy as np
from sklearn import preprocessing, neighbors, model_selection
import pandas as pd

dt = pd.read_csv('breast-cancer-wisconsin.data')
dt.replace('?', -99999, inplace=True)
dt.drop(['id'], 1, inplace=True)

X = np.array(dt.drop(['class'], 1))
y = np.array(dt['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

model = neighbors.KNeighborsClassifier(n_neighbors=15)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)

print(acc)

example_measure = np.array([4,2,1,1,1,2,3,2,1])
example_measure = example_measure.reshape(1, -1)
prediction = model.predict(example_measure)

print(prediction)