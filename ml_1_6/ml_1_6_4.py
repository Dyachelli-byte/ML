# Метрики качества классификации ч.2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

path = r'C:\Users\e.filonova\PycharmProjects\ml_1\ml_1_6\\train.csv'
df = pd.read_csv(path)

# обработка df
df.Sex = df.Sex.map({'male': 0, 'female': 1}).astype(int)
median_male_age = df[(df.Sex == 0) & (df.Age.isnull() is not True)].Age.median()
median_female_age = df[(df.Sex == 1) & (df.Age.isnull() is not True)].Age.median()
df.loc[(df.Age.isnull()) & (df.Sex == 0), 'Age'] = median_male_age
df.loc[(df.Age.isnull()) & (df.Sex == 1), 'Age'] = median_female_age
df.Embarked = df.Embarked.fillna('U')
df.Embarked = df.Embarked.map({'U': 0, 'S': 1, 'C': 2, 'Q': 3}).astype(int)
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

#
target = df.Survived
df.drop(columns=['Survived'], inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=.25, random_state=17)

# 1
"""
Из задания к предыдущему уроку выберите классификатор, который давал наименьшее значение recall,
вычислите для него precision, применив precision_score, и самостоятельно посчитайте F1-меру
(расчеты должны присутствовать). Затем проверьте ответ, используя встроенную функцию.
Сравните полученную f1-меру со значением среднего арифметического полноты и точности.
"""


def f1_score_m(precision, recall, beta) -> float:
    f = ((1 + beta) * precision * recall) / (beta * (precision + recall))
    return f


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_prediction = knn.predict(X_test)
knn_precision = precision_score(y_test, knn_prediction)
knn_recall = recall_score(y_test, knn_prediction)
knn_mean_score = (knn_precision + knn_recall)/2
knn_f1_score = f1_score(y_test, knn_prediction)
print(f'KNN mean score\t{knn_mean_score}\n'
      f'KNN f1_m score(balanced)\t{f1_score_m(knn_precision, knn_recall, 1)}\n'
      f'KNN f1 score\t{knn_f1_score}\n')

"""
В библиотеке sklearn есть удобная функция classification_report, возвращающая precision, recall,
F-меру и количество экземпляров каждого класса в удобном для чтения формате. Также существует 
функция precision_recall_fscore_support, возвращающая те же самые метрики, но в форме массива.
"""
# 2
"""
Для каждого классификатора из предыдущего урока рассчитайте и выведите следующие импортированные
метрики.
"""

print(f'KNN\n{classification_report(y_test, knn_prediction)}\n'
      f'{precision_recall_fscore_support(y_test, knn_prediction)}')

nb = GaussianNB()
nb.fit(X_train, y_train)
nb_prediction = nb.predict(X_test)
print(f'GaussianNB\n{classification_report(y_test, nb_prediction)}\n'
      f'{precision_recall_fscore_support(y_test, nb_prediction)}')

dtc = DecisionTreeClassifier(random_state=17)
dtc.fit(X_train, y_train)
dtc_prediction = dtc.predict(X_test)
print(f'DecisionTreeClassifier\n{classification_report(y_test, dtc_prediction)}\n'
      f'{precision_recall_fscore_support(y_test, dtc_prediction)}')

lr = LogisticRegression(random_state=17)
lr.fit(X_train, y_train)
lr_prediction = lr.predict(X_test)
print(f'LogisticRegression\n{classification_report(y_test, lr_prediction)}\n'
      f'{precision_recall_fscore_support(y_test, lr_prediction)}')

# 3
"""
1 Используя StratifiedKFold, разбейте данные для кросс-валидации по 5-ти блокам (не забывайте во 
всех методах использовать random_state=17).

2 С помощью numpy.logspace разбейте интервал (-1, 2) на 500 значений.

3 С помощью LogisticRegressionCV подберите оптимальный параметр C: установите гиперпараметр Cs 
равным объекту из п.2 (разбиение интервала (-1, 2) отвечает за подбор обратного коэффициента 
регуляризации C); cv равным объекту из п.1 (разбиение для кросс-валидации); scoring равным "roc_auc" 
(отвечает за оптимизацию гиперпараметров на кросс-валидации: метрика, установленная в scoring, 
контролирует, как оценивать модель при каждом из наборе параметров, т.е. показывает, какая метрика 
должна быть наилучшей).

4 Обучите полученную модель на тренировочных данных
"""
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
num_space = np.logspace(-1, 2, 500)
lrCV = LogisticRegressionCV(Cs=num_space, cv=kf, scoring='roc_auc', max_iter=1000)

for i,(train_idx, test_idx) in enumerate(kf.split(df, target)):
    X_train, X_test = df.loc[train_idx], df.loc[test_idx]
    y_train, y_test = target.loc[train_idx], target.loc[test_idx]
    model_lrCV = lrCV.fit(X_train, y_train)

mean_auc_1 = np.mean(model_lrCV.scores_[1], axis=0)

fig, ax = plt.subplots()
ax.plot(num_space, mean_auc_1, label='linear')
plt.show()

# 2
"""
 С помощью метода predict_proba получите вероятности принадлежности объектов тестовой выборки 
 к классам. Постройте график roc_auc для тестовой выборки и выведите значение auc. 
"""
lr_probs = model_lrCV.predict_proba(X_test)
fpr, tpr, treshold = roc_curve(y_test, lr_probs[:, 1])
auc = roc_auc_score(y_test, lr_probs[:, 1])
plt.plot(fpr, tpr, label=f'auc{auc}')
plt.legend(loc=4)
plt.show()
# end = 'end'



