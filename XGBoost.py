# # -*- coding: utf-8 -*-
# """
# Created on Fri Sep 10 14:34:50 2021

# @author: msagming
# """
# import pandas as pd
# import matplotlib.pyplot as plt
# from xgboost import XGBClassifier, plot_tree
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
# from sklearn import preprocessing

# def XGBoost():
#     # Importing the dataset
#     dataset = pd.read_csv('Dataset_Statistics_L5_random.txt', header=None)
#     dataset.columns = ['Column' + str(i) for i in dataset.columns]

#     X = dataset.iloc[:, :24].values
#     y = dataset.iloc[:, 24].values

#     # Normalizing the output data
#     min_max_scaler = preprocessing.MinMaxScaler()
#     y = min_max_scaler.fit_transform(y.reshape(-1, 1)).ravel()

#     # Split data into train and test sets
#     seed = 7
#     test_size = 0.2
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

#     # Fit model on training data
#     model = XGBClassifier(use_label_encoder=False)
#     model.fit(X_train, y_train)  # Remove eval_metric if using older xgboost

#     # Make predictions
#     y_pred = model.predict(X_test)

#     # Confusion matrix
#     from sklearn.metrics import confusion_matrix
#     cm = confusion_matrix(y_test, y_pred)

#     accuracy = cm.trace() / cm.sum()
#     precision = precision_score(y_test, y_pred, average='weighted')
#     recall = recall_score(y_test, y_pred, average='weighted')
#     f1 = f1_score(y_test, y_pred, average='weighted')

#     print("Accuracy: %.2f%%" % (accuracy * 100.00))
#     print("Precision: %.2f%%" % (precision * 100.00))
#     print("Recall: %.2f%%" % (recall * 100.00))
#     print("F-Measure: %.2f%%" % (f1 * 100.00))

#     return accuracy, precision, recall, f1

# if __name__ == '__main__':
#     XGBoost()
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn import preprocessing

def XGBoost(filename):
    # Importing the dataset
    dataset = pd.read_csv(filename, header=None)
    dataset.columns = ['Column' + str(i) for i in dataset.columns]

    X = dataset.iloc[:, :24].values
    y = dataset.iloc[:, 24].values

    # Normalizing the output data
    min_max_scaler = preprocessing.MinMaxScaler()
    y = min_max_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    # Split data into train and test sets
    seed = 7
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Fit model on training data
    model = XGBClassifier(use_label_encoder=False)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    accuracy = cm.trace() / cm.sum()
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Results for dataset: {filename}")
    print("Accuracy: %.2f%%" % (accuracy * 100.00))
    print("Precision: %.2f%%" % (precision * 100.00))
    print("Recall: %.2f%%" % (recall * 100.00))
    print("F-Measure: %.2f%%" % (f1 * 100.00))
    print('-'*40)

    return accuracy, precision, recall, f1

if __name__ == '__main__':
    datasets = [
        'Dataset_Statistics_L5_random.txt',
        'Dataset_Statistics_L10_random.txt',
        'Dataset_Statistics_L15_random.txt'
    ]

    for file in datasets:
        XGBoost(file)

