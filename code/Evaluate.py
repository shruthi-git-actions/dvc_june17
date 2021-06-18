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
import tarfile
import yaml

params = yaml.safe_load(open("home/ubuntu/git_env/dvc_june18/params.yaml"))["featurize"]
no_columns=params["no_columns"]

main_df = dd.read_csv('test_data.csv')
df=main_df[["eventValue","specific_open","specific_close","gateway_change","lost_sight","temp_raise","temp_fall", "came_here"]]

x=df.iloc[:,0:no_columns]
x=x.categorize()
de = DummyEncoder()
X_test = de.fit_transform(x)

Pkl_Filename = "HumanEvent_Model.pkl"  

with open(Pkl_Filename, 'rb') as file:  
    clf = pickle.load(file)

pred_test=clf.predict(X_test)

pred_test_df=dd.from_array(pred_test)
prediction=x.merge(pred_test_df)
prediction.to_csv("out1/prediction_new.csv", single_file = True)


print("End!!")
