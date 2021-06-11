from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import  matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris_dataset = load_iris()
iris_df = pd.DataFrame(iris_dataset['data'], columns=iris_dataset.feature_names)

# Поочередно добавьте к признакам (petal length, petal width) из урока оставшиеся признаки,
# чтобы получилось:
# 1) (sepal length, petal length, petal width);
# 2) (sepal width, petal length, petal width).
# использовала pandas привычнее

iris_df_1 = iris_df.drop(['sepal width (cm)'], axis=1)
iris_df_2 = iris_df.drop(['sepal length (cm)'], axis=1)

print(iris_df_1.describe())
print(iris_df.columns)

# 2
# Теперь посмотрите на трехмерном графике, насколько хорошо данные разделяются
# по каждой из совокупностей трех параметров.

ax = plt.axes(projection='3d')
ax.scatter(iris_df_1['petal length (cm)'],
           iris_df_1['petal width (cm)'],
           iris_df_1['sepal length (cm)'], alpha=.8, c = iris_dataset.target)
plt.show()

ax = plt.axes(projection='3d')
ax.scatter(iris_df_2['petal length (cm)'],
           iris_df_2['petal width (cm)'],
           iris_df_2['sepal width (cm)'], alpha=.8, c = iris_dataset.target)
plt.show()

# 3.
# С помощью функции sklearn.model_selection.train_test_split разделите данные на тренировочный
# и тестовый датасеты и затем, применив библиотечную версию алгоритма
# sklearn.neighbors.KNeighborsClassifier, постройте модель для наборов данных
# iris_dataset_1 и iris_dataset_2 (по умолчанию используйте n_neighbors=5).

x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(iris_df_1, iris_dataset['target'],
                                                            random_state=17)
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(iris_df_1, iris_dataset['target'],
                                                            random_state=17)
knn_1 = KNeighborsClassifier(n_neighbors=5)
knn_2 = KNeighborsClassifier(n_neighbors=5)

knn_model_1 = knn_1.fit(x_train_1, y_train_1)
knn_model_2 = knn_2.fit(x_train_2, y_train_2)

knn_prediction_1 = knn_1.predict((x_test_1))
knn_prediction_2 = knn_2.predict((x_test_2))

print(f'predictrion 1\t{knn_prediction_1}\n'
      f'predictrion 2\t{knn_prediction_2}\n')

# 4.
# Проверьте точность работы обеих моделей, используя встроенную функцию
# sklearn.metrics.accuracy_score. Сравните результат их работы с результатом, полученным на
# наборе данных с двумя признаками (который разбирался в уроке), и укажите ответ.

accuracy_1 = accuracy_score(y_test_1, knn_prediction_1)
accuracy_2 = accuracy_score(y_test_2, knn_prediction_2)

print(f'Accuracy 1\t{accuracy_1}\n'
      f'Accuracy 2\t{accuracy_2}')
# модели показывают одинаковую точность

# 5.
# Постройте модель на данных x_train_1, y_train_1 с гиперпараметром n_neighbors, пробегающим
# значения от 1 до 20 включительно, и укажите значения n_neighbors, которым соответствует
# наиболее высокий результат функции accuracy_score()

def accuracy_score_n(x_train, x_test, y_train, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    knn_predictions = knn.predict(x_test)
    return accuracy_score(y_test, knn_predictions)

accuracy_scores = [accuracy_score_n(x_train_1, x_test_1, y_train_1, y_test_1, k)
                   for k in range(1, 21)]
print(accuracy_scores)

print(f'Индекс наилучшей оценки\t{max(enumerate(accuracy_scores), key=lambda x:x[1])[0]}')