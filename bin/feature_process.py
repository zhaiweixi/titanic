import pandas as pd
import numpy as np
from pandas import Series, DataFrame

data_train = pd.read_csv('../data/Train.csv')

from sklearn.ensemble import RandomForestRegressor

def set_missing_ages(df):
    
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    y = known_age[:, 0]
    X = known_age[:, 1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    predictAges = rfr.predict(unknown_age[:, 1::])

    df.loc[(df.Age.isnull()), 'Age'] = predictAges

    return df, rfr


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'No'
    
    return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)


