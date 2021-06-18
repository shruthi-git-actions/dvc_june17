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

params = yaml.safe_load(open("/home/ubuntu/git_env/dvc_june18/params.yaml"))["prepare"]

file = tarfile.open('/home/ubuntu/data/dataset_small.tar.gz')
file.extractall('./')
split = params["split"]

first_data=pd.read_csv("dataset_small/dataset_small.csv")

df_copy=first_data.copy()
train_set = df_copy.sample(frac=split, random_state=0)
test_set = df_copy.drop(train_set.index)

train_set.to_csv("train_data.csv")
test_set.to_csv("test_data.csv")
