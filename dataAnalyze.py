
#%%
import seaborn as sns
import pandas as pd

train = pd.read_csv('data/first_round_training_data.csv')
feature_name_attr = ['Attribute{0}'.format(i) for i in range(1,4)]
feature_name_param = ['Parameter{0}'.format(i) for i in range(1,11)]

df_param = train[feature_name_param]
h = df_param.hist(bins=25,figsize=(16,16),xlabelsize='10',ylabelsize='10',xrot=-15)
sns.despine(left=True, bottom=True)
[x.title.set_size(12) for x in h.ravel()]
[x.yaxis.tick_left() for x in h.ravel()]

df_attr = train[feature_name_attr]

# %%
