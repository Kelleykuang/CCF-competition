# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
# %%
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


# %%
class PredictModel(object):
    def __init__(self, training_data_path, testing_data_path, submit_data_path):
        self.train = pd.read_csv(training_data_path)
        self.test = pd.read_csv(testing_data_path)
        self.submit = pd.read_csv(submit_data_path)
        self.para_selection = {
            1: ['Parameter' + str(i) for i in range(1, 5)],
            2: ['Parameter' + str(i) for i in range(1, 5)],
            3: ['Parameter' + str(i) for i in range(1, 5)],
            4: ['Parameter' + str(i) for i in range(6, 11)],
            5: ['Parameter' + str(i) for i in range(6, 11)],
            6: ['Parameter' + str(i) for i in range(6, 11)],
            7: ['Parameter' + str(i) for i in range(6, 11)],
            8: ['Parameter' + str(i) for i in range(6, 11)],
            9: ['Parameter' + str(i) for i in range(6, 11)],
            10: ['Parameter' + str(i) for i in range(6, 11)],
        }

        self.xgb_params_set = {
            1: {
                'learning_rate': 0.022,
                'n_estimators': 200,
                'max_depth': 2,
                'min_child_weight': 12,
                'gamma': 0.6,
                'subsample': 0.1,
                'colsample_bytree': 1.0,
                'colsample_bylevel': 0.6,
                'colsample_bynode': 0.4,
            },
            2: {
                'learning_rate': 0.018,
                'n_estimators': 200,
                'max_depth': 1,
                'min_child_weight': 6,
                'gamma': 0.7,
                'subsample': 0.1,
                'colsample_bytree': 0.7,
                'colsample_bylevel': 0.3,
                'colsample_bynode': 0.0,
            },
            3: {
                'learning_rate': 0.031,
                'n_estimators': 200,
                'max_depth': 1,
                'min_child_weight': 16,
                'gamma': 0.6,
                'subsample': 0.1,
                'colsample_bytree': 0.6,
                'colsample_bylevel': 0.5,
                'colsample_bynode': 0.0
            },
            4: {
                'learning_rate': 0.1,
                'n_estimators': 600,
                'max_depth': 2,
                'min_child_weight': 2,
                'gamma': 0.96,
                'subsample': 1.0,
                'colsample_bytree': 0.7,
                'colsample_bylevel': 0.9,
                'colsample_bynode': 0.4,
            },
            5: {
                'learning_rate': 0.12,
                'n_estimators': 480,
                'max_depth': 4,
                'min_child_weight': 15,
                'gamma': 0.97,
                'subsample': 1.0,
                'colsample_bytree': 0.8,
                'colsample_bylevel': 1.0,
                'colsample_bynode': 1.0,
            },
            6: {
                'learning_rate': 0.16,
                'n_estimators': 460,
                'max_depth': 2,
                'min_child_weight': 16,
                'gamma': 0.99,
                'subsample': 0.9,
                'colsample_bytree': 1.0,
                'colsample_bylevel': 0.6,
                'colsample_bynode': 0.7,
            },
            7: {
                'learning_rate': 0.03,
                'n_estimators': 1300,
                'max_depth': 2,
                'min_child_weight': 15,
                'gamma': 0.0,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'colsample_bylevel': 1.0,
                'colsample_bynode': 1.0,
            },
            8: {
                'learning_rate': 0.03,
                'n_estimators': 1300,
                'max_depth': 2,
                'min_child_weight': 18,
                'gamma': 0.0,
                'subsample': 0.8,
                'colsample_bytree': 1.0,
                'colsample_bylevel': 0.8,
                'colsample_bynode': 1.0,
            },
            9: {
                'learning_rate': 0.04,
                'n_estimators': 700,
                'max_depth': 2,
                'min_child_weight': 8,
                'gamma': 0.86,
                'subsample': 0.6,
                'colsample_bytree': 0.8,
                'colsample_bylevel': 0.4,
                'colsample_bynode': 1.0,
            },
            10: {
                'learning_rate': 0.03,
                'n_estimators': 700,
                'max_depth': 2,
                'min_child_weight': 19,
                'gamma': 0.67,
                'subsample': 0.5,
                'colsample_bytree': 1.0,
                'colsample_bylevel': 0.7,
                'colsample_bynode': 1.0,
            },
        }

    def preprocessing(self):
        self.data = self.train.append(self.test).reset_index(drop=True)
        dit = {'Excellent': 0, 'Good': 1, 'Pass': 2, 'Fail': 3}
        self.data['label'] = self.data['Quality_label'].map(dit)

        self.feature_name = ['Parameter{0}'.format(i) for i in range(6, 11)]
        self.attr_name = ['Attribute{0}'.format(i) for i in range(1, 11)]
        self.data[self.feature_name] = np.log1p(self.data[self.feature_name])

        self.data[self.attr_name] = np.log1p(self.data[self.attr_name])
        self.data['sample_weight'] = 1

    def regression(self, train_data, test_data, predict_label, feature_name, attr):
        test_data[predict_label] = 0
        seeds = [20191025, 2019 * 7 + 1023, 4092, 2047, 1021]
        num_model_seed = 5
        for model_seed in range(0, num_model_seed):
            print(model_seed+1)
            skf = KFold(
                n_splits=5, random_state=seeds[model_seed], shuffle=True)
            for train_index, test_index in skf.split(train_data):

                train_x = train_data.loc[train_index][feature_name]
                test_x = train_data.loc[test_index][feature_name]

                train_y = train_data.loc[train_index][attr]
                test_y = train_data.loc[test_index][attr]

                param_set = self.xgb_params_set.get(int(attr[9:]), {})
                xgb_attr_model = xgb.XGBRegressor(
                    # todo 和 n_estimators 成反比
                    learning_rate=param_set.get('learning_rate', 0.02),
                    n_estimators=param_set.get('n_estimators', 200),  # todo

                    max_depth=param_set.get('max_depth', 2),  # todo 太高容易过拟合
                    min_child_weight=param_set.get(
                        'min_child_weight', 10),  # todo

                    gamma=param_set.get('gamma', 0.6),  # todo

                    subsample=param_set.get('subsample', 0.7),  # todo 有不明作用
                    colsample_bytree=param_set.get(
                        'colsample_bytree', 1),  # todo
                    colsample_bylevel=param_set.get(
                        'colsample_bylevel', 1),  # todo
                    colsample_bynode=param_set.get(
                        'colsample_bynode', 1),  # todo

                    objective='reg:squarederror',
                    eval_metric='mae',
                    nthread=-1,
                    verbosity=1,
                )

                xgb_attr_model.fit(train_x, train_y, eval_set=[(test_x, test_y)], verbose=100)

                train_data.loc[test_index,
                               predict_label] = xgb_attr_model.predict(test_x)
                test_data[predict_label] = test_data[predict_label] + \
                    xgb_attr_model.predict(test_data[feature_name])/5

        test_data[predict_label] = test_data[predict_label] / num_model_seed

        return pd.concat([train_data, test_data], sort=True, ignore_index=True)

    def get_predict_label(self):
        
        predict_label_list = []

        for attr in self.attr_name:
            train_index = ~self.data['label'].isnull()
            test_index = self.data['label'].isnull()
            train_data = self.data[train_index].reset_index(drop=True)
            test_data = self.data[~train_index]
            print("Begin training:" + attr + "---------------------------")
            predict_label = 'Predict_Attribute' + attr[9:]
            predict_label_list.append(predict_label)
            self.data = self.regression(
                train_data, test_data, predict_label, self.para_selection[int(attr[9:])], attr)
            print("End training:" + attr + "---------------------------\n")
        newdata = self.data
        return newdata, predict_label_list

    def classifier(self, predict_label_list):
        # newdata, predict_label_list = self.get_predict_label()
        # self.data[self.attr_name] = self.data[predict_label_list]

        tr_index = ~self.data['label'].isnull()
        test_index = self.data['label'].isnull()
        train_data = self.data[tr_index].reset_index(drop=True)
        submit_data = self.data[~tr_index].reset_index(drop=True)
        submit_data[self.attr_name] = submit_data[predict_label_list]

        # model = xgb.XGBClassifier(
        #     learning_rate=0.03,  # 待调整参数
        #     n_estimators=2000,  # 待调整参数
        #     max_depth=5,  # 待调整参数
        #     min_child_weight=5,  # 待调整参数
        #     objective='multi:softprob',
        #     eval_metric='mae',
        #     nthread=-1,
        # )
        # model.fit(X=train_data[predict_label_list], y=train_data['label'])
        # pred = model.predict_proba(submit_data[predict_label_list])
        # # self.get_mae(pred, train_data['label'])
        # print(pred)

        X_train = train_data[self.attr_name].reset_index(drop=True)
        y = train_data['label'].reset_index(drop=True).astype(int)
        X_submit = submit_data[predict_label_list].reset_index(drop=True)

        oof = np.zeros((X_train.shape[0], 4))
        pred = np.zeros((X_submit.shape[0], 4))

        seeds = [20169473, 2011 * 2 + 1066, 20367, 3528, 37293]
        num_model_seed = 5
        for model_seed in range(0, num_model_seed):
            print(model_seed+1)

            oof_cat = np.zeros((X_train.shape[0], 4))
            pred_cat = np.zeros((X_submit.shape[0], 4))

            skf = KFold(
                n_splits=5, random_state=seeds[model_seed], shuffle=True)
            for train_index, test_index in skf.split(train_data):

                train_x = train_data.loc[train_index][predict_label_list]
                test_x = train_data.loc[test_index][predict_label_list]

                train_y = train_data.loc[train_index]['label']
                test_y = train_data.loc[test_index]['label']

                print("Begin training...")
                model = xgb.XGBClassifier(
                    learning_rate=0.14,
                    n_estimators=200,  # done

                    max_depth=9,  # done 可能存在过拟合，需要再次调整
                    min_child_weight=1,  # done
                    gamma=0.46,  # done
                    subsample=0.94,  # done
                    colsample_bytree=1,  # done

                    objective='multi:softmax',
                    num_class=4,
                    eval_metric='merror',
                    nthread=-1,
                    verbosity=1,
                )

                model.fit(train_x, train_y)
                oof_cat[test_index] += model.predict_proba(test_x)
                pred_cat += model.predict_proba(X_submit)/5
                print("End training...")

            oof += oof_cat / num_model_seed
            pred += pred_cat / num_model_seed
            print('logloss', log_loss(pd.get_dummies(y).values, oof_cat))
            print('ac', accuracy_score(y, np.argmax(oof_cat, axis=1)))
            self.get_mae(oof_cat, y)
            print(
                'mae', 1/(1 + np.sum(np.absolute(np.eye(4)[y] - oof_cat))/480))
        return pred

    def get_mae(self, predicted: np.ndarray, standard: pd.Series):
        # pred(n,4) 与 std(n,)
        assert predicted.shape[0] == standard.shape[0]
        n = predicted.shape[0]
        assert predicted.shape == (n, 4)
        assert standard.shape == (n,)

        group_len = 100
        assert n % group_len == 0

        mae_sum = 0

        for i in range(n // group_len):
            p = predicted[i * group_len:(i + 1) *
                          group_len, :].sum(axis=0) / group_len
            counts = standard.iloc[i *
                                   group_len:(i + 1) * group_len].value_counts()
            q = np.array([counts.loc[i] / group_len for i in range(4)])
            mae_sum += np.abs(p - q).sum()
        mae = mae_sum / (n / group_len * 4)
        score = 1 / (1 + 10 * mae)
        print('mae =', mae)
        print('score =', score)

    def submit_ans(self, pred: pd.DataFrame):
        assert pred.shape == (6000, 4)
        n = pred.shape[0]
        sub_data = np.zeros((n // 50, 4))
        for i in range(n // 50):
            sub_data[i, :] = pred[i * 50:(i + 1) * 50, :].sum(axis=0) / 50
        sub = pd.DataFrame(
            data=sub_data,
            columns=['Excellent ratio', 'Good ratio',
                     'Pass ratio', 'Fail ratio']
        )
        sub.index.name = 'Group'
        # print(sub)
        sub.to_csv('submission.csv')

    def run(self):
        self.preprocessing()
        newdata, predict_label_list = self.get_predict_label()
        self.classifier()
        pred = self.classifier(newdata, predict_label_list)
        self.submit_ans(pred)


# %%
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    model = PredictModel(
        training_data_path='data/first_round_training_data.csv', testing_data_path='data/first_round_testing_data.csv', submit_data_path='data/submit_example.csv')
    model.preprocessing()


# %%
    newdata, predict_label_list = model.get_predict_label()


# %%
    pred = model.classifier(predict_label_list)
    print(pred)


# %%
    model.submit_ans(pred)


# %%
