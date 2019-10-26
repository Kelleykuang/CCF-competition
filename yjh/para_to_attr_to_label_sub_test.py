import numpy as np
import pandas as pd
import xgboost as xgb
import util

# 导入数据
train: pd.DataFrame = pd.read_csv('../data/first_round_training_data.csv')
sub: pd.options = pd.read_csv('../data/first_round_testing_data.csv')

# 预处理
train = util.preprocess(train)

para_sub: pd.DataFrame = sub.loc[:, [f'Parameter{i}' for i in range(1, 11)]]
para_sub = np.log(para_sub)
para_sub = (para_sub - para_sub.mean()) / para_sub.std()
sub.loc[:, [f'Parameter{i}' for i in range(1, 11)]] = para_sub

para_selection = {
    1: [6, 7, 8, 9, 10],
    2: [6, 7, 8, 9, 10],
    3: [6, 7, 8, 9, 10],
    4: [6, 7, 8, 9, 10],
    5: [6, 7, 8, 9, 10],
    6: [6, 7, 8, 9, 10],
    7: [6, 7, 8, 9, 10],
    8: [6, 7, 8, 9, 10],
    9: [6, 7, 8, 9, 10],
    10: [6, 7, 8, 9, 10],
}

# ============ tuning for Attribute1 =========
# Fitting 5 folds for each of 14641 candidates, totalling 73205 fits
# -0.7966804100641662 {'colsample_bylevel': 0.8, 'colsample_bynode': 0.0, 'colsample_bytree': 0.5, 'subsample': 0.2}
# ============ tuning for Attribute2 =========
# Fitting 5 folds for each of 14641 candidates, totalling 73205 fits
# -0.7936638670497012 {'colsample_bylevel': 0.0, 'colsample_bynode': 0.0, 'colsample_bytree': 0.6, 'subsample': 1.0}
# ============ tuning for Attribute3 =========
# Fitting 5 folds for each of 14641 candidates, totalling 73205 fits
# -0.7939647518064532 {'colsample_bylevel': 0.9, 'colsample_bynode': 0.0, 'colsample_bytree': 0.8, 'subsample': 0.2}
xgb_params_set = {
    1: {
        'learning_rate': 0.04,
        'n_estimators': 100,
        'max_depth': 2,
        'min_child_weight': 20,
        'gamma': 0.93,
        'subsample': 0.2,
        'colsample_bytree': 0.5,
        'colsample_bylevel': 0.8,
        'colsample_bynode': 0.0,
    },
    2: {
        'learning_rate': 0.01,
        'n_estimators': 350,
        'max_depth': 1,
        'min_child_weight': 23,
        'gamma': 1.0,
        'subsample': 1.0,
        'colsample_bytree': 0.6,
        'colsample_bylevel': 0.0,
        'colsample_bynode': 0.0,
    },
    3: {
        'learning_rate': 0.02,
        'n_estimators': 200,
        'max_depth': 2,
        'min_child_weight': 27,
        'gamma': 0.38,
        'subsample': 0.2,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.9,
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

para_train, attr_train, qual_train, para_test, attr_test, qual_test \
    = util.split_train_test(train, test_rate=0,
                            para_range=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            attr_range=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

attr_pred = pd.DataFrame(
    data={f'Attribute{i}': np.empty(shape=(6000,)) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
)

attr_train_pred = pd.DataFrame(
    data={f'Attribute{i}': np.empty(shape=(6000,)) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
)

for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    param_set = xgb_params_set[i]
    model = xgb.XGBRegressor(
        learning_rate=param_set.get('learning_rate', 0.02),  # todo 和 n_estimators 成反比
        n_estimators=param_set.get('n_estimators', 200),  # todo

        max_depth=param_set.get('max_depth', 2),  # todo 太高容易过拟合
        min_child_weight=param_set.get('min_child_weight', 10),  # todo

        gamma=param_set.get('gamma', 0.6),  # todo

        subsample=param_set.get('subsample', 0.7),  # todo 有不明作用
        colsample_bytree=param_set.get('colsample_bytree', 1),  # todo
        colsample_bylevel=param_set.get('colsample_bylevel', 1),  # todo
        colsample_bynode=param_set.get('colsample_bynode', 1),  # todo

        objective='reg:squarederror',
        eval_metric='mae',
        nthread=-1,
        verbosity=1,
    )
    model.fit(X=para_train.loc[:, [f'Parameter{j}' for j in para_selection[i]]], y=attr_train.loc[:, f'Attribute{i}'])

    attr_train_pred.loc[:, f'Attribute{i}'] = model.predict(
        data=para_train.loc[:, [f'Parameter{j}' for j in para_selection[i]]])

    attr_pred.loc[:, f'Attribute{i}'] = model.predict(data=sub.loc[:, [f'Parameter{j}' for j in para_selection[i]]])

model = xgb.XGBClassifier(
    learning_rate=0.14,  # 0.14 tuned
    n_estimators=200,  # 200 tuned

    max_depth=9,  # done 可能存在过拟合，需要再次调整
    min_child_weight=1,  # done
    gamma=0.46,  # done
    subsample=0.94,  # done

    colsample_bytree=1,  # done
    colsample_bylevel=1,  # done
    colsample_bynode=1,  # done

    objective='multi:softmax',
    num_class=4,
    eval_metric='merror',
    nthread=-1,
    verbosity=1,
)

model.fit(X=attr_train_pred, y=qual_train)
qual_pred = model.predict_proba(data=attr_pred)

util.get_submission(qual_pred)

if __name__ == '__main__':
    pass
