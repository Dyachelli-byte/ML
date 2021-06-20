# Многоклассовая классификация.
"""Посмотрим на примере алгоритма логистической регрессии и метода опорных векторов, как работать с различными методами
многоклассовой классификации."""
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
# 1
"""Вспомните датасет Wine. Загрузите его, разделите на тренировочную и тестовую выборки (random_state=17), используя 
только [9, 11, 12] признаки."""

wine_dataset = load_wine()
X = wine_dataset.data[:, [9,11,12]]
y = wine_dataset.target

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=17)
# Задайте тип кросс-валидации с помощью StratifiedKFold: 5-кратная, random_state=17.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
lr = LogisticRegression(random_state=17, multi_class='ovr')
lr.fit(train_X,train_y)
lr_pred = lr.predict(test_X)
train_cv_score = cross_val_score(lr, train_X, train_y, cv=skf)
test_cv_score = cross_val_score(lr, test_X, test_y, cv=skf)
print(f'train_score\t{train_cv_score.mean()}\n'
      f'test score\t{test_cv_score.mean()}')

# 3
"""
Обучите метод опорных векторов (SVC) с random_state=17 и остальными параметрами по умолчанию.
 Этот метод при мультиклассовой классификации также использует метод "ovr". 
 Сделайте кросс-валидацию (используйте skf) и, как и в предыдущем пункте, выведите среднюю 
 долю правильных ответов на ней. Отдельно выведите долю правильных ответов на тестовой выборке.
 """
clf = SVC(random_state=17)
clf.fit(train_X, train_y)
clf_pred = clf.predict(test_X)
train_clf_score = cross_val_score(clf, train_X, train_y, cv=skf)
test_clf_score = cross_val_score(clf, test_X, test_y, cv=skf)
print(f'train clf score\t{train_clf_score.mean()}\n'
      f'test clf score\t{test_clf_score.mean()}')
# 4
"""
Для предсказаний обеих моделей постройте матрицу ошибок (confusion matrix) и напишите, какие классы 
каждая из моделей путает больше всего между собой.
"""
cm_lr = confusion_matrix(test_y, lr_pred)
cm_clf = confusion_matrix(test_y, clf_pred)
print(f'confusion matrix linear regession\n{cm_lr}\n'
      f'confusion matrix SVC\n{cm_clf}')
# Линейная регрессия может путать 3 класс со 2
# Метод опорных векторов путает 1 класс с 3

# 5
"""
Для каждой модели выведите classification report
"""
print(f'classification report for Linear regression\n{classification_report(test_y, lr_pred)}\n'
      f'classification report for SVC\n{classification_report(test_y, clf_pred)}')