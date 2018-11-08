import numpy as np
import pandas as pd
from sklearn import linear_model

from feature_process import feature_process, set_Cabin_type, set_missing_ages


def train_model(df):
    df = feature_process(data_train)
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.as_matrix()
    y = train_np[:, 0]
    X = train_np[:, 1:]
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)

    return clf


def test_model(df, clf):
    df_test = feature_process(df)
    predictions = clf.predict(df_test)
    return predictions

if __name__ == '__main__':

    data_train = pd.read_csv('../data/train.csv')
    data_train, rfr = set_missing_ages(data_train)
    data_train = set_Cabin_type(data_train)
    data_train = feature_process(data_train)
    clf = train_model(data_train)
    data_test = pd.read_csv('../data/test.csv')

    predict_data = test_model(data_test, clf)

    result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predict_data.astype(np.int32)})

    result.to_csv('../data/prediction.csv')
