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
X_train = X_train * 1000
X_train = np.log(X_train)
# print(X_train)
#%%
y = data[tr_index]['label'].reset_index(drop=True).astype(int)
X_test = data[~tr_index][feature_name].reset_index(drop=True)
X_test = X_test * 1000
X_test = np.log(X_test)
#%%

oof = np.zeros((X_train.shape[0],4))
prediction = np.zeros((X_test.shape[0],4))

#%%
cbt_model_1 = cbt.CatBoostClassifier(iterations=300,learning_rate=0.04,verbose=100,
early_stopping_rounds=1000,task_type='CPU',
loss_function='MultiClass')

cbt_model_2 = cbt.CatBoostClassifier(iterations=400,learning_rate=0.04,verbose=100,
early_stopping_rounds=1000,task_type='CPU',
loss_function='MultiClass')

cbt_model_3 = cbt.CatBoostClassifier(iterations=500,learning_rate=0.04,verbose=100,
early_stopping_rounds=1000,task_type='CPU',
loss_function='MultiClass')
#%%

cbt_model_1.fit(X_train, y, eval_set=(X_train,y))
cbt_model_2.fit(X_train, y, eval_set=(X_train,y))
cbt_model_3.fit(X_train, y, eval_set=(X_train,y))

# oof = cbt_model_2.predict_proba(X_test)

prediction_1 = cbt_model_1.predict_proba(X_test)
prediction_2 = cbt_model_2.predict_proba(X_test)
prediction_3 = cbt_model_3.predict_proba(X_test)

# gc:used to collect garbage 
gc.collect()

# print(prediction[0])
# print('logloss',log_loss(pd.get_dummies(y).values, oof))
# print('ac',accuracy_score(y, np.argmax(oof,axis=1)))
# print('mae',1/(1 + np.sum(np.absolute(np.eye(4)[y] - oof))/480))

sub = test[['Group']]
prob_cols = [i for i in submit.columns if i not in ['Group']]
#print(prob_cols)
#print(enumerate(prob_cols))

for i, f in enumerate(prob_cols):
    sub[f] = (prediction_1[:, i] + prediction_2[:, i] + prediction_3[:, i]) / 3 

for i in prob_cols:
    sub[i] = sub.groupby('Group')[i].transform('mean')
#print(sub)
sub = sub.drop_duplicates()
print(sub)
sub.to_csv('submission.csv',index=False)
sub = np.around(sub, 2)
print(sub)
sub.to_csv('submission_around.csv',index=False)
#%%


'''
iteration = 2000, learning rate = 0.04
0         0         0.238444    0.351801    0.245385    0.164370

iter = 500, lr = 0.04
      Group  Excellent ratio  Good ratio  Pass ratio  Fail ratio
0         0         0.225601    0.340209    0.249001    0.185188
50        1         0.182028    0.266441    0.373095    0.178436
100       2         0.225538    0.300143    0.314767    0.159552


0	0.205671743	0.362045297	0.24972132	0.18256164
1	0.201551188	0.288169692	0.367835246	0.142443874
2	0.2040901	0.295338221	0.310693627	0.189878052

iter = 500 + 700 + 900 ,
0         0         0.229055    0.343287    0.246762    0.180896
50        1         0.180331    0.264443    0.374631    0.180595
100       2         0.222640    0.299370    0.316091    0.161898
150       3         0.246094    0.297348    0.349204    0.107354
200       4         0.181583    0.378660    0.242897    0.196861

iter = 200, lr =0.04
0         0.224414    0.335599    0.252551    0.187437
50        1         0.187743    0.269752    0.367963    0.174542
100       2         0.222332    0.303694    0.305082    0.168892

iter = 700, lr =0.04
0         0         0.229404    0.343603    0.246043    0.180950
50        1         0.179783    0.264118    0.374877    0.181222
100       2         0.222091    0.296099    0.317375    0.164435
'''

