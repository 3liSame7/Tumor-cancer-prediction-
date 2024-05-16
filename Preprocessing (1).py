from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def scaling_data(data_sc):
    # Scaling data
    print("StandardScaler: 1, MinMaxScaler: 2")
    choose = input()
    if choose == '1':
        scaling = StandardScaler()
        data_sc = pd.DataFrame(scaling.fit_transform(data_sc))

    elif choose == '2':
        scaling = MinMaxScaler()
        data_sc = pd.DataFrame(scaling.fit_transform(data_sc))
    else:
        print("Error")
        return 1

    return data_sc


def cleaning(data):
    # encode diagnosis column M: 1, B: 0
    label_encoder = LabelEncoder()

    # drop unnecessary column
    data.drop('Index', axis=1, inplace=True)

    y_data = label_encoder.fit_transform(data.iloc[:, 30])
    data = data.iloc[:, 0:30]
    y_data = pd.Series(y_data)

    data = data[(data != 0).all(1)]
    data = data.dropna(how='any', axis=0)

    data['diagnosis'] = y_data
    data.drop_duplicates(keep=False, inplace=True)
    return data
