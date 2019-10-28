#%%
import numpy as np
import catboost as cbt
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import gc
import math
import time
from tqdm import tqdm
import datetime
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from datetime import datetime,timedelta
import warnings
import os
import pandas as pd

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None
pd.options.display.max_rows = None

train = pd.read_csv('data/first_round_training_data.csv')
test = pd.read_csv('data/first_round_testing_data.csv')
submit = pd.read_csv('data/submit_example.csv')

# permute as natural index
data = train.append(test).reset_index(drop=True)

dit = {'Excellent':0,'Good':1,'Pass':2,'Fail':3}
data['label'] = data['Quality_label'].map(dit)

# feature engineering
feature_name = ['Parameter{0}'.format(i) for i in range(6,11)]

tr_index = ~data['label'].isnull()  # test data don't have quality label
# The code below is just used to get train data X and label Y
X_train = data[tr_index][feature_name].reset_index(drop=True)
X_train = np.log(X_train)
# print(X_train)
#%%
y = data[tr_index]['label'].reset_index(drop=True).astype(int)
X_test = data[~tr_index][feature_name].reset_index(drop=True)
#%%

oof = np.zeros((X_train.shape[0],4))
prediction = np.zeros((X_test.shape[0],4))

#%%
cbt_model_1 = cbt.CatBoostClassifier(iterations=2000,learning_rate=0.04,verbose=100,
early_stopping_rounds=1000,task_type='CPU',
loss_function='MultiClass')

cbt_model_2 = cbt.CatBoostClassifier(iterations=5000,learning_rate=0.04,verbose=100,
early_stopping_rounds=1000,task_type='CPU',
loss_function='MultiClass')

cbt_model_3 = cbt.CatBoostClassifier(iterations=1000,learning_rate=0.04,verbose=100,
early_stopping_rounds=1000,task_type='CPU',
loss_function='MultiClass')
#%%

cbt_model_2.fit(X_train, y, eval_set=(X_train,y))
oof = cbt_model_2.predict_proba(X_test)

prediction = cbt_model_2.predict_proba(X_test)

# gc:used to collect garbage 
gc.collect()

print(prediction[0])
print('logloss',log_loss(pd.get_dummies(y).values, oof))
print('ac',accuracy_score(y, np.argmax(oof,axis=1)))
print('mae',1/(1 + np.sum(np.absolute(np.eye(4)[y] - oof))/480))

sub = test[['Group']]
prob_cols = [i for i in submit.columns if i not in ['Group']]
#print(prob_cols)
#print(enumerate(prob_cols))

for i, f in enumerate(prob_cols):
    sub[f] = prediction[:, i]

for i in prob_cols:
    sub[i] = sub.groupby('Group')[i].transform('mean')
#print(sub)
sub = sub.drop_duplicates()
print(sub)
sub.to_csv('submission.csv',index=False)