# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 13:35:30 2021

@author: msagming
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# ----------------------------
# 1. Load and prepare dataset
# ----------------------------
dataset = pd.read_csv('Dataset_Statistics_L10_random.txt', header=None)
dataset.columns = ['Column' + str(i) for i in range(dataset.shape[1])]

x = dataset.iloc[:, :24].values
y = dataset.iloc[:, 24].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# ----------------------------
# 2. Define hyperparameter search space
# ----------------------------
hyperparameter_space = {
    'max_depth': hp.quniform('max_depth', 3, 18, 1),
    'gamma': hp.uniform('gamma', 1, 9),
    'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    'n_estimators': hp.quniform('n_estimators', 50, 200, 1),
    'seed': hp.quniform('seed', 0, 10, 1)
}

# ----------------------------
# 3. Objective function for optimization
# ----------------------------
def objective(space):
    clf = xgb.XGBClassifier(
        n_estimators=int(space['n_estimators']),
        max_depth=int(space['max_depth']),
        gamma=space['gamma'],
        reg_alpha=int(space['reg_alpha']),
        reg_lambda=space['reg_lambda'],
        min_child_weight=int(space['min_child_weight']),
        colsample_bytree=space['colsample_bytree'],
        seed=int(space['seed']),
        use_label_encoder=False,
        eval_metric='auc'  # Set directly here to avoid set_params
    )

    eval_set = [(X_test, y_test)]

    # Using early_stopping_rounds requires proper validation set & metric
    clf.fit(
        X_train, y_train,
        eval_set=eval_set,
        early_stopping_rounds=10,
        verbose=False
    )

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"[INFO] Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    return {'loss': -acc, 'status': STATUS_OK}

# ----------------------------
# 4. Run hyperparameter optimization
# ----------------------------
trials = Trials()

best_hyperparams = fmin(
    fn=objective,
    space=hyperparameter_space,
    algo=tpe.suggest,
    max_evals=500,
    trials=trials
)

print("\n Best hyperparameters found:")
print(best_hyperparams)