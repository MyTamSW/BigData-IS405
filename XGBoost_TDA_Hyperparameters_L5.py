# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 14:34:50 2021

@author: msagming
"""
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def XGBoost():
    # Importing the dataset
    dataset = pd.read_csv('Dataset_Statistics_L5_random.txt', header = None)
    dataset.columns = 'Column' + dataset.columns.astype(str)
    
    x = dataset.iloc[:, :24].values
    y = dataset.iloc[:, 24].values
    #Nonormalising the output data
    min_max_scaler = preprocessing.MinMaxScaler()
    y = min_max_scaler.fit_transform(y.reshape(-1,1))
    
    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    # fit model no training data
    model = XGBClassifier(use_label_encoder=False, n_estimators=188, reg_lambda=0.7301, reg_alpha=74, min_child_weight=4, max_depth=17, gamma=1.053, colsample_bytree=0.928, seed=0)
    model.fit(X_train, y_train, eval_metric='logloss')
    # make predictions for test data
    y_pred = model.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    # Compute the accuracy
    accuracy = cm.trace()/cm.sum()
    print("Accuracy: %.2f%%" % (accuracy*100.00))
    
    # Compute the precision
    precision = precision_score(y_test, y_pred)
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
