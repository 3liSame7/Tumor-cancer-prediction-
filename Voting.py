from sklearn.ensemble import VotingClassifier
from SVM import svm
from LogisticRegression import log
from RandomForest import forest
from DecisionTree import tree
from Naive import naive
from sklearn.metrics import accuracy_score

voting = VotingClassifier(estimators=[('logistic regression', log), ('svm', svm), ('random forest', forest), ('decision tree', tree), ('naive', naive)], voting='hard')


def voting_model(x_train, y_train):
    voting.fit_transform(x_train, y_train)


def predict_voting(x_test):
    y_pred = voting.predict(x_test)

    b_m = []

    for i in y_pred:

        if i == 0.0:
            b_m.append('B')

        elif i == 1.0:
            b_m.append('M')
    print('Voting Predication:', b_m)


def report_voting(x_test, y_test):
    print("Voting Accuracy Score:", accuracy_score(y_test, voting.predict(x_test)))
