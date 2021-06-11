# Урок 8. Деревья решений.
# В уроках мы разбирали на небольшом наборе данных деревья решений без настройки гиперпараметров.
# Здесь предлагаем вам рассмотреть работу классификатора на более сложном датасете,
# применив настройку гиперпараметров.
# 1
# На датасете из предыдущего урока - load_wine() - обучите дерево решений (DecisionTreeClassifier).
# Примечание: при установке гиперпараметров модели и разбиении на тренировочный и тестовый
# датасеты используйте random_state=17 для воспроизводимости результатов.
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz

wine_dataset = datasets.load_wine()
X = wine_dataset.data
y = wine_dataset.target
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=17)

dtc = DecisionTreeClassifier(random_state=17)
dtc_model = dtc.fit(train_X, train_y)
# 2
# Отобразите дерево с помощью библиотеки graphviz.

def print_graph(data):

    dot_data = tree.export_graphviz(data, out_file=None,
                                    feature_names=wine_dataset.feature_names,
                                    class_names=wine_dataset.target_names,
                                    filled=True)
    return graphviz.Source(dot_data)

print_graph(dtc_model).view()

# 3
# Используя полученную модель, сделайте прогноз для тестовой выборки данных и, пользуясь встроенной
# функцией score(), определите точность модели.

dtc_predict = dtc.predict(test_X)
accuracy = dtc.score(test_X, test_y)
print(f'Accuracy\t{accuracy}')

# 4
# Постройте модель, указав гиперпараметр max_features равным 2 (так же указав random_state=17) и,
# сделав прогноз на тестовых данных, определите, стала ли модель работать точнее.
# Примечание: гиперпараметр max_features показывает модели, сколько различных признаков необходимо
# использовать каждый раз при принятии решения о разделении. То есть если, допустим, у вас имеются
# 50 признаков, а max_features=10, то в каждом узле случайным образом выбираются 10 признаков,
# которые будут использоваться для поиска лучшего разделения.

dtc_2 = DecisionTreeClassifier(max_features=2, random_state=17)
dtc_model_2 = dtc_2.fit(train_X, train_y)
dtc_predict_2 = dtc_2.predict(test_X)
accuracy_2 = dtc_2.score(test_X, test_y)
print(f'Accuracy_2\t{accuracy_2}')
# Модель получилась менее точной
# 5
# Теперь постройте граф дерева решений, используя graphviz.

print_graph(dtc_model_2).view()

# Модель работает точнее, чем без настройки гиперпараметров, но по графу можно увидеть,
# что глубина дерева увеличилась, при этом в нескольких листах присутствует только по одному объекту,
# из-за чего на тестовом наборе данных дерево могло несколько потерять обобщающую способность.
# Так как главная задача модели - способность обобщаться на новые данные, то для подбора оптимальных
# гиперпараметров надо пожертвовать небольшой порцией данных, чтобы на ней во время обучения
# проверять качество модели.
# Часто для этого используют кросс-валидацию: модель обучается K раз на (K-1) подвыборках исходной
# выборки, а на одной подвыборке проверяется (каждый раз на разной). Получаются K оценок качества
# модели, которые обычно усредняются, давая среднюю оценку.
# Кросс-валидация применяется для подбора гиперпараметров модели в методе GridSearchCV():
# для каждой уникальной совокупности значений параметров проводится кросс-валидация и выбирается
# лучшее сочетание параметров.

# 6
# Используя обученное в предыдущем задании дерево решений, настройте параметры max_depth и
# max_features на 5-кратной валидации с помощью GridSearchCV. Для этого в функцию GridSearchCV
# передайте параметры (tree, tree_params, cv=5).
# Примечание: tree здесь - не обученная модель, а объект, который инкапсулирует алгоритм.
# Например, в уроке мы его обозначали как dtc.

from sklearn.model_selection import GridSearchCV, cross_val_score

tree_params = {
    'max_depth': range(1, 6),
    'max_features': range(1, 10)
}
tree_grid = GridSearchCV(dtc_2, tree_params, cv=5)

# 7
# С помощью метода tree_grid.fit() постройте модель на тренировочных данных и выведите лучшее
# сочетание параметров с помощью метода tree_grid.best_params_

tree_grid.fit(train_X,train_y)
print(tree_grid.best_params_)

# 8
# С помощью полученной модели сделайте прогноз - predict - для тестовой выборки и выведите долю 
# верных ответов, проверив точность модели, используя функцию accuracy_score.

from sklearn.metrics import  accuracy_score

dtc_predict_3 = tree_grid.predict(test_X)
accuracy_3 = accuracy_score(test_y, dtc_predict_3)
print(f'Accuracy_3\t{accuracy_3}')


