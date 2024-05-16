from sklearn import svm


SVM = svm.SVC(kernel='linear')


def svm_model(x_train, y_train):
    SVM.fit(x_train, y_train)
    print("SVM Score:", SVM.score(x_train, y_train))


def predict_svm(y_test):
    print("SVM Predication", SVM.predict(y_test))


