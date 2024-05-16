from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

naive = GaussianNB()

def naive_model(x_train, y_train):
    naive.fit(x_train, y_train)
    print('Naive score: ', naive.score(x_train, y_train))

def predict_naive(x_test):
    print('Naive Prediction: ', naive.predict(x_test))

def report_naive(x_test, y_test):
    print("Naive Accuracy Score:", accuracy_score(y_test, naive.predict(x_test)))
