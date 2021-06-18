import dask_ml
import pandas as pd
import dask.dataframe as dd
from dask_ml.preprocessing import DummyEncoder
import numpy as np
from dask.distributed import Client
from sklearn.ensemble import RandomForestClassifier
import joblib
import pickle
import dask
from dask_ml.metrics import accuracy_score
import os, uuid
import json
import yaml

params = yaml.safe_load(open("home/ubuntu/git_env/dvc_june18/params.yaml"))["featurize"]
no_columns = params["no_columns"]
main_df = dd.read_csv('/home/shruthi/Shruthi_Tasks/GiTAction/dvc_june10/dvc_demo/train_data.csv')
print(main_df)
df=main_df[["eventValue","specific_open","specific_close","gateway_change","lost_sight","temp_raise","temp_fall", "came_here"]]
print(no_columns)

x=df.iloc[:,0:no_columns]
y_train=df.iloc[:,-1]

x=x.categorize()
de = DummyEncoder()
X_train = de.fit_transform(x)

X_train.to_csv("out/X_train.csv", single_file = True)
x.to_csv("out/x.csv", single_file = True)
y_train.to_csv("out/y_train.csv", single_file = True)
