from sklearn import svm
from sklearn.metrics import accuracy_score

svm = svm.SVC(kernel='linear', random_state=0)


def svm_model(x_train, y_train):
    svm.fit(x_train, y_train)
    print("SVM Score:", svm.score(x_train, y_train))


def predict_svm(x_test):
    print("SVM Prediction:", svm.predict(x_test))


def report_svm(x_test, y_test):
    print("SVM Accuracy Score:", accuracy_score(y_test, svm.predict(x_test)))
