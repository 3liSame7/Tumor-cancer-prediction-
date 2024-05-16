# import numpy as np
# import pandas as pd
# from sklearn import linear_model, metrics
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
#
# l = linear_model.LinearRegression()
# # poly = PolynomialFeatures(degree=1)
#
#
# def linear_regression_model(x_train, y_train):
#     # x_poly = poly.fit_transform(x_train)
#     l.fit(x_train, y_train)
#
# # poly.fit_transform
# def predict_linear_regression(x_test):
#     y_train_predicted = linear_model.predict(x_test)
#     return y_train_predicted
#
#
# def linear_report(x_test, y_test):
#     print('Co-efficient of linear regression', linear_model.coef_)
#     print('Intercept of linear regression model', linear_model.intercept_)
#     x_test = pd.DataFrame(x_test)
#
#     true_player_value = np.asarray(y_test)[0]
#     predicted_player_value = predict_linear_regression(x_test)[0]
#
#     print('True value for the first player in the test set in millions is : ' + str(true_player_value))
#     print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value))
#
# # predicting on training data-set
#
# # predicting on test data-set
# # prediction = poly_model.predict(poly_features.fit_transform(X_test))
