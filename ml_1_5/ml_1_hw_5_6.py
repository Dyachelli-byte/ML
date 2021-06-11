import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

# 1.
# Посмотрите, что включает в себя новый датасет Wine, где собраны результаты химического анализа
# вин, выращенных в одном регионе Италии тремя различными культиваторами.
# В наборе каждый из объектов имеет 13 числовых признаков.

# Изучите попарные графики признаков (в уроках мы рассматривали их с помощью функции
# pd.plotting.scatter_matrix() ) и выберите такие два признака, используя которые, на ваш взгляд,
# можно будет лучше всего разделить данные по трем классам. Помните о том, что модель гауссовского
# наивного Байеса использует для предсказаний среднее и разброс значений признаков относительно
# среднего, поэтому вероятнее всего, лучшими признаками будут те, значения которых на графике
# разбросаны примерно одинаково от среднего значения, но как можно меньше накладываются на
# признаки других классов.

wine_dataset = datasets.load_wine()
print('_'*100)
print('Датасет wine\n')
print(wine_dataset)
# Из датасета в датафрейм, чтобы удобнее было работать
df_wine = pd.DataFrame(wine_dataset['data'], columns=wine_dataset.feature_names)
scat_matrix = pd.plotting.scatter_matrix(df_wine, c=wine_dataset['target'], figsize=(25, 25), marker='o', hist_kwds={'bins': 20}, s=40, alpha=.8)
plt.show()
print('_'*100)
# 2.
# Разбейте данные на тренировочный и тестовый датасеты (при разбиении используйте параметр
# random_state=17 для воспроизводимости результатов) и постройте модель на двух выбранных признаках.
# Используя встроенную функцию score(), проверьте точность работы модели. Если score() меньше
# 0.88..., выберите по графику другие два признака и постройте модель на них. Укажите номера
# признаков, которые вы использовали (помните, что индекс массива признаков начинается с 0).

# У модели GaussianNB есть метод predict_proba(test), который возвращает вероятности принадлежности
# каждого объекта из test к каждому из классов.

X = df_wine[['proline', 'od280/od315_of_diluted_wines']]
y = wine_dataset.target
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=17)
nb = GaussianNB()
nb_model = nb.fit(train_X, train_y)
accuracy = nb.score(test_X,test_y)
print(f'Качество модели для признаков proline/od280/od315_of_diluted_wines\t{accuracy}')
X = df_wine[['proline', 'hue']]
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=17)
nb_2 = GaussianNB()
nb_model_2 = nb_2.fit(train_X, train_y)
accuracy_2 = nb_2.score(test_X,test_y)
print(f'Качество модели для признаков proline/hue\t{accuracy_2}')
print('_'*100)
# 3.
# Еще раз обучите модель на признаках с номерами 11, 12, предварительно снова разбив данные на
# тренировочные и тестовые (с random_state=17). С помощью функции predict_proba() посмотрите,
# какие вероятности были вычислены для каждого из классов, и выведите эти вероятности для
# объекта x_test[0].
X = wine_dataset.data[:,11:13]
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=17)
nb_3 = GaussianNB()
nb_model_3 = nb_3.fit(train_X, train_y)
pb_3 = nb_3.predict_proba(test_X)
accuracy_3 = nb_3.score(test_X,test_y)
print(f'Вероятности принадлежности к классу test_x[0]{pb_3[0]}')
print(f'Качество модели для признаков '
      f'{"".join([wine_dataset.feature_names[11], "/",wine_dataset.feature_names[12]])}'
      f'\t{accuracy_3}')