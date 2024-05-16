from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

forest = RandomForestClassifier(n_estimators=10, random_state=0, criterion='entropy')


def random_forest_model(x_train, y_train):
    forest.fit(x_train, y_train)
    print('RandomForest score: ', forest.score(x_train, y_train))

def predict_random_forest(x_test):
    print("Random forest Prediction", forest.predict(x_test))

def report_random_forest(x_test, y_test):
    print("RandomForest Accuracy Score:", accuracy_score(y_test, forest.predict(x_test)))
