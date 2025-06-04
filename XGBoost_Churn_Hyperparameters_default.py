# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 14:34:50 2021

@author: msagming
"""
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import preprocessing

def XGBoost():
    
    
    # Importing the dataset
    dataset = pd.read_csv('Churn_tuned_dataset.csv', header = None)
    dataset.columns = 'Column' + dataset.columns.astype(str)

    X = dataset.iloc[:, :-1].values  # tất cả cột ngoại trừ cột cuối cùng
    y = dataset.iloc[:, -1].values
     
    #Nonormalising the output data
    min_max_scaler = preprocessing.MinMaxScaler()
    y = min_max_scaler.fit_transform(y.reshape(-1,1))
    
    # split data into train and test sets
    seed = 7
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    # fit model no training data
    model = XGBClassifier()
    model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    # Compute the accuracy
    accuracy = cm.trace()/cm.sum()
    print("Accuracy: %.2f%%" % (accuracy*100.00))
    
    # Compute the precision
    precision = precision_score(y_test, y_pred, average='weighted')
    print("Precision: %.2f%%" % (precision*100.00))
    
    # Compute the recall
    recall = recall_score(y_test, y_pred, average='weighted')
    print("Recall: %.2f%%" % (recall*100.00))
    
    # Compute the precision
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F-Measure: %.2f%%" % (f1*100.00))
    
    return accuracy, precision, recall, f1

if __name__ == '__main__':
    XGBoost()
