import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(train: pd.DataFrame, log: str = 'log', normalize=True):
    p = train.loc[:, 'Parameter1':'Attribute10']
    if log == 'log':
        p = np.log(p)
    elif log == 'log1p':
        p = np.log1p(p)
    else:
        raise ValueError("'log' must be 'log' or 'log1p'")
    if normalize:
        p = (p - p.mean()) / p.std()  # 标准化
    train.loc[:, 'Parameter1':'Attribute10'] = p
    d = {'Excellent': 0, 'Good': 1, 'Pass': 2, 'Fail': 3}
    q = train.loc[:, 'Quality_label']
    train.loc[:, 'Quality_label'] = q.map(d).astype(int)
    return train


def split_train_test(train: pd.DataFrame, test_rate: float,
                     para_range=[6, 7, 8, 9, 10],  # 5,6 重复
                     attr_range=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    if test_rate != 0:
        train, test = train_test_split(train, test_size=test_rate)
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        para_train = train.loc[:, ['Parameter%d' % i for i in para_range]]
        attr_train = train.loc[:, ['Attribute%d' % i for i in attr_range]]
        qual_train = train.loc[:, 'Quality_label']
        para_test = test.loc[:, ['Parameter%d' % i for i in para_range]]
        attr_test = test.loc[:, ['Attribute%d' % i for i in attr_range]]
        qual_test = test.loc[:, 'Quality_label']
        return para_train, attr_train, qual_train, para_test, attr_test, qual_test
    else:
        para_train = train.loc[:, ['Parameter%d' % i for i in para_range]]
        attr_train = train.loc[:, ['Attribute%d' % i for i in attr_range]]
        qual_train = train.loc[:, 'Quality_label']
        return para_train, attr_train, qual_train, None, None, None


def get_mae(predicted: np.ndarray, standard: pd.Series):
    # pred(n,4) 与 std(n,)
    assert predicted.shape[0] == standard.shape[0]
    n = predicted.shape[0]
    assert predicted.shape == (n, 4)
    assert standard.shape == (n,)

    group_len = 100
    assert n % group_len == 0

    mae = 0
    for i in range(n // group_len):
        p = predicted[i * group_len:(i + 1) * group_len, :].sum(axis=0) / group_len
        counts = standard.iloc[i * group_len:(i + 1) * group_len].value_counts()
        q = np.array([counts.loc[i] / group_len for i in range(4)])
        mae += np.abs(p - q).sum() / 100
    score = 1 / (1 + 10 * mae)
    print('mae =', mae)
    print('score =', score)


def get_submission(pred: np.ndarray):
    assert pred.shape == (6000, 4)
    n = pred.shape[0]
    sub_data = np.zeros((n // 50, 4))
    for i in range(n // 50):
        sub_data[i, :] = pred[i * 50:(i + 1) * 50, :].sum(axis=0) / 50
    sub = pd.DataFrame(
        data=sub_data,
        columns=['Excellent ratio', 'Good ratio', 'Pass ratio', 'Fail ratio']
    )
    sub.index.name = 'Group'
    print(sub)
    sub.to_csv('submission.csv')


if __name__ == '__main__':
    data = np.abs(np.random.randn(6000 * 4)).reshape((6000, 4))
    print(data)
    get_submission(data)
