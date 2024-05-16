import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import pandas as pd


def scaling_data(data_sc):
    # Scaling data
    scaling = StandardScaler()
    data_sc = pd.DataFrame(scaling.fit_transform(data_sc))
    return data_sc


def cleaning(data):
    # Drop Unnecessary Column
    data.drop('Index', axis=1, inplace=True)

    # Encode Diagnosis Column M: 1, B: 0
    label_encoder = LabelEncoder()

    # 1d Array
    y_data = label_encoder.fit_transform(data.iloc[:, 30])

    data = data.iloc[:, 0:30]

    imp = SimpleImputer(missing_values=np.NaN, strategy='mean')
    data = pd.DataFrame(imp.fit_transform(data))
    data.drop_duplicates(keep=False, inplace=True)


    # Features Extraction
    pca = PCA(n_components=4, random_state=0)
    data = pca.fit_transform(data)
    data = pd.DataFrame(data,
                        columns=['F1', 'F2', 'F3', 'F4'])

    # Drop Outliers
    q1 = data.quantile(0.05)
    q3 = data.quantile(0.95)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    data = data[(data.iloc[:, 0:4] > lower) & (data.iloc[:, 0:4] < upper)]
    data['diagnosis'] = y_data

    # Replace NAN With Mean Of Column
    imp = SimpleImputer(missing_values=np.NaN, strategy='mean')
    data = pd.DataFrame(imp.fit_transform(data))

    return data
