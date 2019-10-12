import numpy as np
import pandas as pd
import catboost as cbt
from sklearn.metrics import accuracy_score, roc_auc_score,log_loss
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
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None
pd.options.display.max_rows = None

train = pd.read_csv('data/first_round_training_data.csv')
test = pd.read_csv('data/first_round_testing_data.csv')
submit = pd.read_csv('data/submit_example.csv')
data = train.append(test).reset_index(drop=True)
dit = {'Excellent':0,'Good':1,'Pass':2,'Fail':3}
data['label'] = data['Quality_label'].map(dit)

feature_name = ['Parameter{0}'.format(i) for i in range(5, 11)]
tr_index = ~data['label'].isnull()
X_train = data[tr_index][feature_name].reset_index(drop=True)
y = data[tr_index]['label'].reset_index(drop=True).astype(int)
X_test = data[~tr_index][feature_name].reset_index(drop=True)

print(X_train.shape,X_test.shape)
oof = np.zeros((X_train.shape[0],4))
prediction = np.zeros((X_test.shape[0],4))

cbt_model = cbt.CatBoostClassifier(iterations=2000,learning_rate=0.04,verbose=100,
early_stopping_rounds=1000,task_type='CPU',
loss_function='MultiClass')
cbt_model.fit(X_train, y ,eval_set=(X_train,y))
oof = cbt_model.predict_proba(X_test)
prediction = cbt_model.predict_proba(X_test)
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
sub.to_csv('submission.csv',index=False)