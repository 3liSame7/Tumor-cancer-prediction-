from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', random_state=0)


def decision_tree_model(x_train, y_train):
    tree.fit(x_train, y_train)
    print("Decision Tree Score:", tree.score(x_train, y_train))


def predict_decision_tree(X_test):
    print("Decision Tree Prediction", tree.predict(X_test))

def report_decision_tree(x_test, y_test):
    print("Decision Tree Accuracy Score:", accuracy_score(y_test, tree.predict(x_test)))
