from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

log = LogisticRegression(random_state=0, solver='liblinear')


def logistic_regression_model(x_train, y_train):
    log.fit(x_train, y_train)
    print("Logistic Regression Score:", log.score(x_train, y_train))


def predict_logistic_regression(y_test):
    print("Logistic Regression Predication:", log.predict(y_test))


def report_logistic_regression(x_test, y_test):
    print("Logistic Regression Accuracy Score:", accuracy_score(y_test, log.predict(x_test)))