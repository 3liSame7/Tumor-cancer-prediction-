# import libraries and packages
import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm


# from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def scaling_data(data_sc):
    choose = input("For normalization Scaler choose 1 and for Standard Scaler choose 2")

    if choose == '1':
        scaling = StandardScaler()
        data_sc = pd.DataFrame(scaling.fit_transform(data_sc))
    elif choose == '2':
        scaling = MinMaxScaler()
        data_sc = pd.DataFrame(scaling.fit_transform(data_sc))
    else:
        print("Error")

    return data_sc


# load the data
data = pd.read_csv('Tumor Cancer Prediction_Data.csv')

# drop unnecessary column
data.drop('Index', axis=1, inplace=True)

# encode diagnosis column M: 1, B: 0
LabelEncoder_Y = LabelEncoder()
Y_data = LabelEncoder_Y.fit_transform(data.iloc[:, 30])

data = data.iloc[:, 0:30]

# drop outliers
Q1 = data.quantile(0.05)
Q3 = data.quantile(0.95)

IQR = Q3 - Q1

Lower = Q1 - 1.5 * IQR
Upper = Q3 + 1.5 * IQR

data = data[(data.iloc[:, 0:30] > Lower) & (data.iloc[:, 0:30] < Upper)]

#rows will be dropped
print(data.isna().sum())
data.dropna(how='any', inplace=True, )

# Scaling data
# data = scaling_data(data)
Y_data = pd.Series(Y_data)

# print(Y_data.head(10))

data['diagnosis'] = Y_data
print(data.head(20))

# print(Y_data.head(20))
# print(data['diagnosis'].value_counts())

# print(data.head(10))

X_data = data.iloc[:, 0:30]
Y_data = data.iloc[:, 30]

# print(X_data.shape)
# print(Y_data.shape)
# print(type(X_data))
# print(type(Y_data))
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=0, stratify=Y_data)
print(X_train.head(10))
# print(Y_train.head(10))

log = LogisticRegression(random_state=0, solver='liblinear', C=10)
log.fit(X_train, Y_train)
print(log.score(X_train, Y_train))

svm = svm.SVC(kernel='linear')
svm.fit(X_train, Y_train)
print(svm.score(X_train, Y_train))

# print(classification_report(Y_test, log.predict(X_test)))
# print(classification_report(Y_test, svm.predict(X_test)))
#
print(accuracy_score(Y_test, svm.predict(X_test)))
print(accuracy_score(Y_test, log.predict(X_test)))


x = [0.009924, 11.37, 18.89, 72.17, 0.03416, 0.08713, 0.1118, 0.05008, 0.02399, 0.01376, 0.02173,
     0.01395, 0.06994, 0.2013, 0.05955, 459.3, 0.2656, 0.3267, 0.09708, 1.974, 1.954, 17.49, 0.006538, 0.07529,
     0.002928, 12.36, 26.14, 79.29, 396, 0.06203]
x = np.array([x]).reshape(1, 30)

# x_test=np.expand_dims(x_test, axis=1)
x = pd.DataFrame(x, columns=['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13',
                             'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'F21', 'F22', 'F23', 'F24', 'F25',
                             'F26', 'F27', 'F28', 'F29', 'F30'])
x = pd.DataFrame(x)
print(x)
y_test = log.predict(x)
print(y_test)
y_test = svm.predict(x)
print(y_test)

# [0.01666, 14.22, 23.12, 94.37, 0.05113, 0.1075, 0.1533, 0.2413, 0.1981, 0.1166, 0.06618, 0.1354, 0.1446, 0.2384,
# 0.07542, 762.4, 0.286, 0.5166, 0.9327, 2.11, 2.112, 31.72, 0.00797, 0.8488, 0.01172, 15.74, 37.18, 106.4, 609.9,
# 0.1772]>>>>1
#
#
# [0.01843, 11.8, 16.58, 78.99, 0.05628, 0.1091, 0.1385, 0.17, 0.1659, 0.04649, 0.07415, 0.03633, 0.103,
# 0.2678, 0.07371, 591.7, 0.3197, 0.5774, 0.4092, 1.426, 2.281, 24.72, 0.005427, 0.4504, 0.004635, 13.74, 26.38,
# 91.93, 432, 0.1865]>>>>>1
#
# [0.009924, 11.37, 18.89, 72.17, 0.03416, 0.08713, 0.1118, 0.05008, 0.02399, 0.01376, 0.02173,
# 0.01395, 0.06994, 0.2013, 0.05955, 459.3, 0.2656, 0.3267, 0.09708, 1.974, 1.954, 17.49, 0.006538, 0.07529,
# 0.002928, 12.36, 26.14, 79.29, 396, 0.06203] >>>0>>>188
