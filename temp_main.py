import numpy as np
import pandas as pd
import catboost as cbt
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import gc
import math
import time
from tqdm import tqdm
import datetime
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import warnings
import os
from sklearn.utils import shuffle

class PredictModel(object):
    def __init__(self, training_data_path, testing_data_path, submit_data_path):
        self.train = pd.read_csv(training_data_path)
        self.test = pd.read_csv(testing_data_path)
        self.submit = pd.read_csv(submit_data_path)

    def preprocessing(self):
        self.data = self.train.append(self.test).reset_index(drop=True)
        dit = {'Excellent': 0, 'Good': 1, 'Pass': 2, 'Fail': 3}
        self.data['label'] = self.data['Quality_label'].map(dit)

        self.feature_name = ['Parameter{0}'.format(i) for i in range(5,11)]
        self.attr_name = ['Attribute{0}'.format(i) for i in range(1, 11)]
        self.data[self.feature_name] = np.log1p(self.data[self.feature_name])

        self.data[self.attr_name] = np.log1p(self.data[self.attr_name])
        self.data['sample_weight'] = 1

    def regression(self, train_data, test_data, predict_label, feature_name, attr):
        test_data[predict_label] = 0
        seeds = [19970412, 2019 * 2 + 1024, 4096, 2048, 1024]
        num_model_seed = 5
        for model_seed in range(0, num_model_seed):
            print(model_seed+1)
            skf = KFold(n_splits=5, random_state=seeds[model_seed], shuffle=True)
            for train_index, test_index in skf.split(train_data):

                train_x = train_data.loc[train_index][feature_name]
                test_x = train_data.loc[test_index][feature_name]

                train_y = train_data.loc[train_index][attr]
                test_y = train_data.loc[test_index][attr]

                lgb_attr_model = lgb.LGBMRegressor(boosting_type="gbdt", metric='rmse', num_leaves=31,  reg_alpha=10, reg_lambda=5, max_depth=7,n_estimators=500, subsample=0.7, colsample_bytree=0.4,   subsample_freq=2, min_child_samples=10, learning_rate=0.05)
                lgb_attr_model.fit(train_x, train_y, eval_set=[(test_x, test_y)], eval_metric='mae',
                                   categorical_feature=feature_name, sample_weight=train_data.loc[train_index]  ['sample_weight'], verbose=100)

                train_data.loc[test_index,
                               predict_label] = lgb_attr_model.predict(test_x)
                test_data[predict_label] = test_data[predict_label] + lgb_attr_model.predict(test_data[feature_name])

        test_data[predict_label] = test_data[predict_label] / num_model_seed

        return pd.concat([train_data, test_data], sort=True, ignore_index=True)
    
    def get_predict_label(self):
        train_index = ~self.data['label'].isnull()
        test_index = self.data['label'].isnull()
        train_data = self.data[train_index].reset_index(drop=True)
        test_data = self.data[~train_index]
        predict_label_list = []
        for attr in attr_name:
            print("Begin training" + attr + "---------------------------")
            predict_label = 'Predict_Attribute' + attr[9:]
            predict_label_list.append(predict_label)
            new_data = self.regression(train_data, test_data, predict_label, feature_name, attr)
            print("End training" + attr + "---------------------------\n")
        return new_data
    
    def classifier(self):
        pass

    def submit(self):
        pass

    def run(self):
        self.preprocessing()
        self.get_predict_label()
        self.classifier()
        self.submit()

if __name__ == "__main__":
    model = PredictModel(
        training_data_path='data/first_round_training_data.csv', testing_data_path='data/first_round_testing_data.csv', submit_data_path='data/submit_example.csv')
    model.run()