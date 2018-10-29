import pandas as pd
import numpy as np
from pandas import Series, DataFrame

data_train = pd.read_csv('../data/Train.csv')

import matplotlib.pyplot as plt

fig = plt.figure()
fig.set(alpha=0.2)

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({'survived': Survived_1, 'unsurvived': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title('pclass distribute')
plt.xlabel('class')
plt.ylabel('num')
plt.show()


