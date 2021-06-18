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
import dvc.api

params = yaml.safe_load(open("home/ubuntu/git_env/dvc_june18/params.yaml"))["featurize"]
with dvc.api.open(
        'train_data.csv',
        repo='https://github.com/shruthi-git-actions/dvc_june17.git',
        remote='remote_storage',
        encoding='utf-8'
        ) as fd:
    main_df_c=pd.read_csv(fd)
main_df = dd.read_csv(main_df_c)

df=main_df[["eventName", "eventValue", "specific_open", "specific_close", "gateway_change", "lost_sight", "temp_raise", "temp_fall", "came_here","human_event"]]

no_columns=len(df.columns)
x=df.iloc[:,0:no_columns]
y_train=df.iloc[:,-1]

x=x.categorize()
de = DummyEncoder()
X_train = de.fit_transform(x)

X_train.to_csv("out/X_train.csv", single_file = True)
x.to_csv("out/x.csv", single_file = True)
y_train.to_csv("out/y_train.csv", single_file = True)
