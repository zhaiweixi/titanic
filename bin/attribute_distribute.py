# coding: utf-8
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

data_train = pd.read_csv('../data/Train.csv')
#data_train.info()

import matplotlib.pyplot as plt

fig = plt.figure()
fig.set(alpha = 0.2) # 设定图标的alpha参数

plt.subplot2grid((2, 3), (0, 0)) # 在一张大图中分列几个小图
data_train.Survived.value_counts().plot(kind='bar') # 柱状图
plt.title('survive distribute')
plt.ylabel('num')

plt.subplot2grid((2, 3), (0, 1)) # 在一张大图中分列几个小图
data_train.Pclass.value_counts().plot(kind='bar') # 柱状图
plt.title('num')
plt.ylabel('pclass')

plt.subplot2grid((2, 3), (0, 2)) # 在一张大图中分列几个小图
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel('age')
plt.grid(b=True, which='major', axis='y')
plt.title('age distribute')

plt.subplot2grid((2, 3), (1, 0), colspan=2) # 在一张大图中分列几个小图
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel('age')
plt.ylabel('density')
plt.title('pclass age distribute')
plt.legend(('class1', 'class2', 'class3'), loc='best')

plt.subplot2grid((2, 3), (1, 2)) # 在一张大图中分列几个小图
data_train.Embarked.value_counts().plot(kind='bar') # 柱状图
plt.title('embarked')
plt.ylabel('pclass')


plt.show()

