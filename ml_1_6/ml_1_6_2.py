import pandas as pd
import numpy as np
path = r'C:\Users\e.filonova\PycharmProjects\ml_1\6\\train.csv'
df = pd.read_csv(path)

# survival 0 = No, 1 = Yes
# pclass Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
# sibsp Количество братьев и сестер / супругов на борту Титаника
# parch Количество родителей / детей на борту Титаника
# ticket Ticket number
# fare Passenger fare Пассажирский тариф
# embarked Port of Embarkation Порт погрузки C = Cherbourg, Q = Queenstown, S = Southampton

# 1.1
# Выкачайте тренировочный датасет Titanic - train.csv - с сайта kaggle. С помощью функции
# pd.read_csv() загрузите данные в датафрейм. Выведите первые 20 строк и проанализируйте данные:
# какие колонки присутствуют (более конкретная информация по ним есть на сайте kaggle),
# каким образом в них обозначены данные и какие типы данных используются (используйте pandas.dtypes).

print(df.info())
print(df.head(20))
# 1.2
# Проверьте, имеются ли пропущенные значения в колонках, и выведите сумму всех пропущенных
# значений в каждой из колонок.
print(f'Сумма пропущенных значений в колонках\n{df.isnull().sum()}')

# 1.3
# Замените все пропущенные значения колонки Age на медианы в зависимости от пола человека: т.е.
# если пол человека в строке с пропущенным значением "male", заменяете пропущенное значение возраста
# на медиану по всем известным возрастам мужчин, и наоборот. Выведите медианы возраста в зависимости
# от пола. Пропущенные значения колонок Cabin и Embarked замените на U (Unknown).

median_male_age = df[(df.Sex == 'male') & (df.Age.isnull() is not True)].Age.median()
median_female_age = df[(df.Sex == 'female') & (df.Age.isnull() is not True)].Age.median()

df.Age = np.where((df.Age.isnull()) & (df.Sex == 'male'),
                  median_male_age,
                  df.Age)
df.Age = np.where((df.Age.isnull()) & (df.Sex == 'female'),
                  median_female_age,
                  df.Age)
df.Cabin = np.where(df.Cabin.isnull(), 'U', df.Cabin)
df.Embarked = np.where(df.Embarked.isnull(), 'U', df.Embarked)

print(f'Медиана возраста мужчин\t{median_male_age}\n'
      f'Медиана возраста женщин\t{median_female_age}')

# 1.4
# Выведите возраст пассажиров с PassengerID = [6, 20]. Убедитесь, что заполнены все пропущенные
# значения (воспользуйтесь функцией df.isnull() ).

print(df[(df.PassengerId == 6) | (df.PassengerId == 20)][['Name', 'Sex', 'Age']])
print(df.info())

# 1.5
# В колонке Sex замените значения на 0, если пол "male", и на 1, если "female".
# В колонке Embarked замените параметры "U", "S", "C", "Q" на 0, 1, 2, 3 соответственно.
# Отбросьте колонки PassengerId, Name, Ticket, Cabin.
# Выведите первые 20 строк получившегося набора данных.

df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
df.Sex = np.where(df.Sex == 'male', 0, 1)
def replace_ctg(embarked) -> int:
    category = 0
    if embarked == 'S': category = 1
    elif embarked == 'C': category = 2
    elif embarked == 'Q': category = 3
    return category

vect_ctg = np.vectorize(replace_ctg)
df.Embarked = vect_ctg(df.Embarked)
print(df.head(20))

# 1.6
# Приступим к построению классификаторов. Условимся, что, если функция или объект модели имеют
# параметр random_state, то устанавливаем его равным 17 в каждом из случаев.
# Разделите данные на тренировочный и тестовый датасеты, установив размер тестового как 0.25
# (первая колонка Survived является целевой, поэтому необходимо сначала ее отделить от признаков).

target = df.Survived
df.drop(columns=['Survived'], inplace=True)

from sklearn.model_selection import train_test_split
import sklearn.metrics

X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=.25, random_state=17)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_prediction = knn.predict(X_test)

knn_precision = sklearn.metrics.precision_score(y_test, knn_prediction)
knn_recall = sklearn.metrics.recall_score(y_test, knn_prediction)

print(f'\nknn recall\t{knn_recall}\n'
      f'knn precision\t{knn_precision}')
# мало ложных срабатываний, но много неучтенных

# лучше чтобы были живые среди мертвых(precision) или мертвые среди (recall) живых?
# для спасателей мертвые среди живых
# для новостей живые среди мертвых, т.е чтобы количество жертв не увеличивалось, а уменьшалось

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_prediction = nb.predict(X_test)

nb_precision = sklearn.metrics.precision_score(y_test, nb_prediction)
nb_recall = sklearn.metrics.recall_score(y_test, nb_prediction)

print(f'\nnb recall\t{nb_recall}\n'
      f'nb precision\t{nb_precision}')

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=17)
dtc.fit(X_train, y_train)
dtc_prediction = dtc.predict(X_test)

dtc_precision = sklearn.metrics.precision_score(y_test, dtc_prediction)
dtc_recall = sklearn.metrics.recall_score(y_test, dtc_prediction)

print(f'\ndtc recall\t{dtc_recall}\n'
      f'dtc precision\t{dtc_precision}')

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=17)
lr.fit(X_train, y_train)
lr_prediction = lr.predict(X_test)

lr_precision = sklearn.metrics.precision_score(y_test, lr_prediction)
lr_recall = sklearn.metrics.recall_score(y_test, lr_prediction)

print(f'\nlr recall\t{lr_recall}\n'
      f'lr precision\t{lr_precision}')

# В каждом из пунктов следующего теста выберите, что важнее максимизировать: точность (precision)
# или полноту (recall). Выпишите ответы.
#
# 1 Вероятность того, что при определенной поломке самолета он сможет долететь до пункта
# назначения (1 - долетел, 0 - не долетел).

# Ответ: точность, лучше наверняка знать что самолет долетел, чем немколько пропущенных катастроф

# 2 Предсказание, представляет ли человек опасность, по анализу психического состояния (1 -
# представляет опасность, 0 - не представляет опасности).

# Ответ: полнота, лучше отправить на доп проверку, чем вероятность несчастного случая

# 3 Предсказание ухода клиента (1 - клиент ушел, 0 - остался).

# Ответ: полнота, в данном случае несколько неверно определенных как ушедные не повредят

# 4 Выявление рака на основе медицинских показателей (1 - болен раком, 0 - здоров).

#Ответ: полнота, ничего страшного, если пациент пройдет еще несколько анализов и в итоге диагноз
# окажется не верным, чем упустить больного

# 5 Предсказание летальности при наблюдаемой мутации (1 - выживание, 0 - летальный исход).

# Ответ: полнота

# 6 Определение важности происшествия для экстренных служб (1 - важно, 0 - неважно).

# Ответ: полнота, лучше несколько раз ложное срабатывание, чем хотя бы 1 пропущенное

# 7 Окупятся ли вложения в бизнес (1 - окупятся, 0 - не окупятся).

# Ответ: зависит от того насколько свобоны средства, если потеря данных средств не скажется
# сильно, то можно использовать параметр полноты, в том случае если средства ограниченны,
# тогда лучше точность