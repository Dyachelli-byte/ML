
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from scipy.stats import skew
from scipy.stats import kurtosis
import statsmodels.api as sm

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
#print(wine_dataset)
# Из датасета в датафрейм, чтобы удобнее было работать
df_wine = pd.DataFrame(wine_dataset['data'], columns=wine_dataset.feature_names)
# смотрим признаки при помощи scatter plot
# scat_matrix = pd.plotting.scatter_matrix(df_wine, c=wine_dataset['target'], figsize=(25, 25),
#                                         marker='o', hist_kwds={'bins': 20}, s=40, alpha=.8)
# plt.show()
# однако, я не очень доверяю глазу в таких оценках и если "лучшими признаками будут те, значения
# которых на графике разбросаны примерно одинаково от среднего значения"

# 2.
# Разбейте данные на тренировочный и тестовый датасеты (при разбиении используйте параметр
# random_state=17 для воспроизводимости результатов) и постройте модель на двух выбранных признаках.
# Используя встроенную функцию score(), проверьте точность работы модели. Если score() меньше
# 0.88..., выберите по графику другие два признака и постройте модель на них. Укажите номера
# признаков, которые вы использовали (помните, что индекс массива признаков начинается с 0).

# У модели GaussianNB есть метод predict_proba(test), который возвращает вероятности принадлежности
# каждого объекта из test к каждому из классов.
# экперемент №1 по ассиметрии


fig, ax = plt.subplots(2,3,figsize = (8,25))
sm.qqplot(df_wine['od280/od315_of_diluted_wines'], fit=True, line='45', ax=ax[0][0])
sm.qqplot(df_wine['proline'], fit=True, line='45', ax=ax[0][0], color='red')
ax[0][0].set_title('od280/od315_of_diluted_wines/proline',fontsize=10)

sm.qqplot(df_wine['hue'], fit=True, line='45', ax=ax[0][1])
sm.qqplot(df_wine['proline'], fit=True, line='45', ax=ax[0][1], color='red')
ax[0][1].set_title('hue/proline',fontsize=10)

sm.qqplot(df_wine['nonflavanoid_phenols'], fit=True, line='45', ax=ax[0][2])
sm.qqplot(df_wine['proline'], fit=True, line='45', ax=ax[0][2], color='red')
ax[0][2].set_title('nonflavanoid_phenols/proline',fontsize=10)

sm.qqplot(df_wine['nonflavanoid_phenols'], fit=True, line='45', ax=ax[1][0])
sm.qqplot(df_wine['od280/od315_of_diluted_wines'], fit=True, line='45', ax=ax[1][0], color='red')
ax[1][0].set_title('nonflavanoid_phenols/od280/od315_of_diluted_wines',fontsize=10)

sm.qqplot(df_wine['proanthocyanins'], fit=True, line='45', ax=ax[1][1])
sm.qqplot(df_wine['od280/od315_of_diluted_wines'], fit=True, line='45', ax=ax[1][1], color='red')
ax[1][1].set_title('proanthocyanins/od280/od315_of_diluted_wines',fontsize=10)
plt.show()


def check(feature_1, feature_2) -> float:
    x = df_wine[[feature_1, feature_2]]
    y = wine_dataset.target
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=17)
    nb = GaussianNB()
    nb.fit(train_x, train_y)
    nb.predict(test_x)
    return nb.score(test_x, test_y)


columns_name = list(df_wine.columns.values)
columns_name.append('name')

df_sum = pd.DataFrame(columns=columns_name)
df_sum.name = df_wine.columns
df_sum.set_index('name', inplace=True)
print('_' * 100)

dict_2 = {}
for col in df_sum.columns:
    dict_2[col] = {}
    for idx in df_sum[col].index:
        if idx != col:
            ch = round(check(col, idx), 3)
            if idx not in dict_2:
                if idx not in dict_2[col]:
                    dict_2[col][idx] = ch


# for k in dict_2:
#    if dict_2[k] > 0.95:
#        print(k, dict_2[k])
for k in dict_2:
    for h in dict_2[k]:
        print(k,h,'\t',dict_2[k][h])
df_3 = pd.DataFrame(dict_2)
for col in df_3.columns:
    print(df_3[col].sort_values(ascending=True))
#for k in dict_2:
#    if 0.95 >= dict_2[k] > 0.90:
 #       print(k, dict_2[k])
# 3.
# Еще раз обучите модель на признаках с номерами 11, 12, предварительно снова разбив данные на
# тренировочные и тестовые (с random_state=17). С помощью функции predict_proba() посмотрите,
# какие вероятности были вычислены для каждого из классов, и выведите эти вероятности для
# объекта x_test[0].
