import sys
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm

import mlflow
import mlflow.sklearn

import os


mlflow.set_tracking_uri("http://10.185.33.168:5000/")

#mlflow.set_tracking_uri("http://localhost:5000/")

data = pd.read_csv("data/dataset.csv")

targets = data[["score_bin"]]
#print(targets)
data.drop([ "score_bin",'score_multi','score_norm'], inplace=True, axis=1)
data=data.iloc[:,2:]

x_train, x_test, y_train, y_test = train_test_split(
    data.values,
    targets.values.ravel(),
    test_size=0.3,
    random_state=2021,
    stratify=targets.values,
)
experiment_id = mlflow.set_experiment("Projet_Mlops")

n_estimators=20
max_depth=5
learning_rate=0.1
num_leaves=12

with mlflow.start_run():
    
    model = lgb.LGBMClassifier(learning_rate=learning_rate, max_depth=max_depth,
    min_child_weight=2, min_split_gain=0.01,n_estimators=n_estimators,num_leaves=num_leaves,
    reg_alpha=0.1, reg_lambda=0.1, subsample=0.9)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    print(auc)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_leaves", num_leaves)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("auc", auc)

    mlflow.sklearn.log_model(model, "model")

