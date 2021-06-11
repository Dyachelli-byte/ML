from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris_dataset = load_iris()
print(iris_dataset.keys())
print(iris_dataset['DESCR'][:177])
print(iris_dataset['target_names'])
print(iris_dataset['feature_names'])
print(iris_dataset['target'].shape)

import pandas as pd

iris_dataframe = pd.DataFrame(iris_dataset['data'], columns=iris_dataset.feature_names)
scat_matrix = pd.plotting.scatter_matrix(iris_dataframe, c=iris_dataset['target'], figsize=(5,5),
                                         marker='o', hist_kwds={'bins': 20}, s=40, alpha=.8)
plt.show()
iris_dataframe_simple = pd.DataFrame(iris_dataset.data[:, 2:4],
                                     columns=iris_dataset.feature_names[2:4])
scat_matrix_2 = pd.plotting.scatter_matrix(iris_dataframe_simple, c=iris_dataset['target'],
                                          figsize=(5,5), marker='o', hist_kwds={'bins': 20}, s=40,
                                          alpha=.8)
plt.show()
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(iris_dataset.data[:, 2:4], iris_dataset.target,
                                                    random_state=0)
import numpy as np

x_train_concat = np.concatenate((train_X, train_y.reshape(112,1)),axis=1)
x_test_concat = np.concatenate((test_X, test_y.reshape(38,1)),axis=1)
print(f'x_train_shape\t{x_train_concat.shape}\n'
      f'x_test_shape\t{x_test_concat.shape}')
print(pd.DataFrame(x_train_concat).head(5))
# в последнем столбце теперь присутствуют метки класса
import math

def euclidean_distance(data1, data2):
    distance = 0
    for i in range(len(data1) - 1):
        distance+= (data1[i] - data2[i]) ** 2
    return math.sqrt(distance)

def get_neighbors(train, test, k=1):
    distance = [(train[i][-1], euclidean_distance(train[i], test))
                for i in range(len(train))]
    distance.sort(key=lambda elem: elem[1])
    neighbors = [distance[i][0] for i in range(k)]
    return neighbors

def prediction(neighbors):
    count = {}
    for instance in neighbors:
        if instance in count:
            count[instance] +=1
        else:
            count[instance] = 1
    target = max(count.items(), key=lambda x: x[1])[0]
    return target

def accuracy(test, test_prediction):
    correct = 0
    for i in range(len(test)):
        if test[i][-1] == test_prediction[i]:
            correct += 1
    return (correct / len(test))

predictions = []
for x in range(len(x_test_concat)):
    neighbors = get_neighbors(x_train_concat, x_test_concat[x], k=5)
    result = prediction(neighbors)
    predictions.append(result)
accuracy_ = accuracy(x_test_concat, predictions)
print(f'Accuracy\t{accuracy_}')
# чз sklearn
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_X, train_y)
knn_predictions = knn.predict(test_X)
print(knn_predictions)

from sklearn.metrics import  accuracy_score
accuracy = accuracy_score(test_y, knn_predictions)
print(f'Accuracy\t{accuracy}')