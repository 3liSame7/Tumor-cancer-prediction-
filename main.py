# import libraries and packages
import seaborn as sns
from matplotlib import pyplot as plt
from RandomForest import *
from sklearn.model_selection import train_test_split
from Preprocessing import *
from LogisticRegression import *
from SVM import *
from Voting import *
from Save_And_Load import *
from DecisionTree import *
from Naive import *

# Read Csv File
data = pd.read_csv('cancertestnew.csv')

data = cleaning(data)

#visualize the correlation
plt.figure(figsize=(10,10))
sns.heatmap(data.iloc[:,0:4].corr(),annot=True,fmt='.0%')
plt.show()
sns.countplot(data.iloc[:, 4],label='count')
plt.show()

X_test = data.iloc[:, 0:4]
# X_test = scaling_data(X_test)

Y_test = data.iloc[:, 4]
# print(Y_data)
# Split Data To Train And Test
# X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=0, stratify=Y_data)

# Create Models
log = load('logistic_regression_model')
svm = load('svm_model')
# naive = load('naive_model')
forest = load('random_forest_model')
tree = load('decision_tree_model')
voting = load('voting_model')

# svm_model(X_data, Y_data)
# save(svm)
#
# logistic_regression_model(X_data, X_data)
# save(log)
#
# decision_tree_model(X_data, Y_data)
# save(tree)
#
# naive_model(X_data, Y_data)
# save(naive)
#
# random_forest_model(X_data, Y_data)
# save(forest)
#
# voting_model(X_data, Y_data)
# save(voting)

def convert(y_pred):
    b_m = []

    for i in y_pred:

        if i == 0.0:
            b_m.append('B')

        elif i == 1.0:
            b_m.append('M')
    print(b_m)

print('SVM accuracy score: ', accuracy_score(Y_test, svm.predict(X_test)))
svm.predict(X_test)
convert(svm.predict(X_test))
print(svm.predict(X_test))

# print('Naive accuracy score: ', accuracy_score(Y_test, naive.predict(X_test)))
# naive.predict(X_test)
# convert(naive.predict(X_test))
# print(naive.predict(X_test))

print('Logistic Regression accuracy score: ', accuracy_score(Y_test, log.predict(X_test)))
log.predict(X_test)
convert(log.predict(X_test))
print(log.predict(X_test))

print('Random Forest accuracy score: ', accuracy_score(Y_test, forest.predict(X_test)))
forest.predict(X_test)
convert(forest.predict(X_test))
print(forest.predict(X_test))

print('Decision Tree accuracy score: ', accuracy_score(Y_test, tree.predict(X_test)))
tree.predict(X_test)
convert(tree.predict(X_test))
print(tree.predict(X_test))



print("Voting Accuracy Score:", accuracy_score(Y_test, voting.predict(X_test)))
convert(voting.predict(X_test))
print(voting.predict(X_test))

# print("Enter row to predict it")

# predict = [float(x) for x in input().split()]
# predict = np.array([predict]).reshape(1, 30)