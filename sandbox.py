# -*- coding: utf-8 -*-
# %%
# import zenitai
# import matplotlib.pyplot as plt
# from zenitai.transform.woe import WoeTransformer, WoeTransformerRegularized
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zenitai
from zenitai.utils.utils import generate_train_data
from zenitai.transform import WoeTransformer, grouping, statistic, cat_features_alpha_logloss

df = generate_train_data()
X = df.drop("target", axis=1)
y = df["target"]
wt = WoeTransformer()
# print(wt)

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# %%
# %load_ext autoreload
# %aimport zenitai
# %autoreload 1

# %%
# wt.fit(X, y)
# with open('zenitai/temp/wt_original.pkl', 'wb') as f:
#   pickle.dump(wt, f)


# %%
# %%prun -l 10
wt = WoeTransformer()
wt.fit(X, y)


# %%
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
km = KMeans(2)

# %%
df.columns

# %%
letts = df['letters']
cat_floats = df['many_floats_w_cat']
target = df['target']

# %%
from sklearn.base import BaseEstimator, TransformerMixin
class MeanLabelEncoder(BaseEstimator, TransformerMixin):
    ''' Класс
    '''
    pass


# %%
# # %%prun -l 4
def encode_categories(x):
    num_mask = pd.to_numeric(x, errors='coerce').notna()
    if num_mask.sum() > 0:
        numerics = x[num_mask]
        strings = x[~num_mask]
        min_val = float(numerics.min())
        uniqs = set(strings)
        repl = {cat: min_val*(i+2) * (-np.sign(min_val)) for i, cat in enumerate(uniqs)}
    else:
        uniqs = set(x)
        repl = {cat: i for i, cat in enumerate(uniqs)}

    return  pd.to_numeric(x.replace(repl))

encode_categories(pd.Series(list(range(-150, -50)) + ['a', 'b']*100))

# %%
list(cat_floats[pd.to_numeric(cat_floats, errors='coerce').isna()].unique())

# %%
X_copy = df.drop('target', axis=1).copy()
X_copy = X_copy.apply(encode_categories, axis=0)
X_copy_norm = pd.DataFrame(StandardScaler().fit_transform(X_copy), columns=X_copy.columns)

# %%
km.fit(X_copy_norm)

# %%
y_km = km.labels_

# %%

lr = LogisticRegression()
lr.fit(X_copy, y_km)
preds = lr.predict_proba(X_copy)[:, 1]
roc_auc_score(y_km, preds)

# %%
plt.hist(preds)

# %%
df_km = pd.concat([pd.DataFrame(X_copy), ], axis=1)

# %%
df_km_woe = wt.fit_transform(X, pd.Series(y_km, name='target'))

# %%
len(wt.predictors), len(X.columns)

# %%
pd.Series(y_km, name='target').head(500)

# %%
sns.pairplot(df_km_woe.head(500),
             hue=pd.Series(y_km, name='target').head(500))

# %%
