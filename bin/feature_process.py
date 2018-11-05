# coding: utf-8
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing


def set_missing_ages(df):
    
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    y = known_age[:, 0]
    X = known_age[:, 1:]
    # n_jobs=-1 means as more as number of CPU
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    predictAges = rfr.predict(unknown_age[:, 1::])

    df.loc[(df.Age.isnull()), 'Age'] = predictAges

    return df, rfr


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'No'
    
    return df


if __name__ == '__main__':
    data_train = pd.read_csv('../data/Train.csv')
    data_train, rfr = set_missing_ages(data_train)
    data_train = set_Cabin_type(data_train)
    
    dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
    
    df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Cabin', 'Embarked', 'Sex', 'Pclass', 'Ticket', 'Name'], axis=1, inplace=True)
    
    scaler = preprocessing.StandardScaler()
    # print type(df['Age'])
    age_scale_param = scaler.fit(df[['Age']])
    age_scaled = scaler.fit_transform(df[['Age']], age_scale_param)
    
    df['Age_scaled'] = scaler.fit_transform(df[['Age']], age_scale_param)
    fare_scale_param = scaler.fit(df[['Fare']])
    df['fare_scaled'] = scaler.fit_transform(df[['Fare']], fare_scale_param)

    print df
    df.info()

