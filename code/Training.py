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

params = yaml.safe_load(open("home/ubuntu/git_env/dvc_june18/params.yaml"))["Training"]
n_est = params["n_estimators"]
n_j= params["n_jobs"]

with dvc.api.open(
        'out/X_train.csv',
        repo='https://github.com/shruthi-git-actions/dvc_demo.git',
        remote='remote_storage',
        encoding='utf-8'
        ) as fd1:
    X_train_c=pd.read_csv(fd1)
X_train = dd.read_csv(X_train_c)
with dvc.api.open(
        'out/x.csv',
        repo='https://github.com/shruthi-git-actions/dvc_demo.git',
        remote='remote_storage',
        encoding='utf-8'
        ) as fd2:
    x_c=pd.read_csv(fd2)

x =dd.read_csv(x_c)
with dvc.api.open(
        'out/y_train.csv',
        repo='https://github.com/shruthi-git-actions/dvc_demo.git',
        remote='remote_storage',
        encoding='utf-8'
        ) as fd3:
    y_train_c=pd.read_csv(fd3)
y_train =dd.read_csv(y_train_c)

client = Client(processes=False)             # create local cluster

clf = RandomForestClassifier(n_estimators=n_est, n_jobs=n_j)
with joblib.parallel_backend("dask", scatter=[X_train, y_train]):
 clf.fit(X_train, y_train)


# In[ ]:



Pkl_Filename = "HumanEvent_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(clf, file)


# In[ ]:


pred_train=clf.predict(X_train)


# In[ ]:


pred_train_df=dd.from_array(pred_train)
#pred_train_df=pred_train_df.to_frame()
training=x.merge(pred_train_df)


# In[ ]:




training.to_csv("out/training_new.csv", single_file = True)


# In[ ]:


print(accuracy_score(np.array(y_train),np.array(pred_train)))
print ("end training!")
