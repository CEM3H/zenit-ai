# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     formats: ipynb,python//py:percent
#     notebook_metadata_filter: all,-language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   toc:
#     base_numbering: 1
#     nav_menu: {}
#     number_sections: false
#     sideBar: true
#     skip_h1_title: false
#     title_cell: Table of Contents
#     title_sidebar: Contents
#     toc_cell: false
#     toc_position: {}
#     toc_section_display: true
#     toc_window_display: false
# ---

# %% ExecuteTime={"end_time": "2020-07-03T17:02:53.905951Z", "start_time": "2020-07-03T17:02:53.858766Z"}
from pathlib import Path
import os
os.chdir('/Users/stepankadocnikov/Documents/GitHub/woe-transformer')
print(os.getcwd())


# %% ExecuteTime={"end_time": "2020-07-03T17:02:54.811316Z", "start_time": "2020-07-03T17:02:54.783465Z"}
import json
import warnings
import math
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from woeTransformer_class import WoeTransformer, WoeTransformerRegularized

from collections import defaultdict
from tqdm.notebook import tqdm
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification

# %load_ext autoreload
# %aimport utils
# %aimport woeTransformer_class
# %autoreload 1

# %%

# %%
X, y = make_classification(5000, 20, 5, 0, weights=(0.95, 0.05), shift=5, scale=2)
X = pd.DataFrame(X)
for i, f in enumerate(X.columns):
    if i <= 9:
        X[f] = X[f].astype(int)
    if i & 2 == 0:
        ind = np.random.choice(range(len(X)), 500, replace=False)
        X.loc[ind, f] = np.nan
    X[f] = X[f].astype(str)
y = pd.Series(y, name='target')
X[0].name

# %% ExecuteTime={"end_time": "2020-07-03T19:04:39.772315Z", "start_time": "2020-07-03T19:04:39.769279Z"}
n_seeds = 1
alphas = [0, 0.0001, 0.001, 0.01, 0.02, .03]


# %%
vanilla = WoeTransformer()
vanilla.fit(X[[0]], y)

# %% ExecuteTime={"end_time": "2020-07-03T19:13:35.250809Z", "start_time": "2020-07-03T19:12:55.764500Z"} scrolled=false
regul = WoeTransformerRegularized(alphas=alphas, n_seeds=n_seeds)
regul.fit(X.iloc[:,[0]], y)



# %%
a = [[0, 0.203948491746367],
     [0.0001, 0.20394840230562705],
     [0.001, 0.2039206201054458],
     [0.01, 0.2036637928958567],
     [0.02, 0.20353968262545227],
     [0.03, 0.20347955300751042]]
np.min(a, axis=0), np.argmin(a, 0), np.argmin(a, 0)[1], a[np.argmin(a, 0)[1]][0]

# %%
class GroupedPredictor(pd.DataFrame):
        """
        Вспомогательный класс для удобства доступа к некоторым данным
        """
        def get_predictor(self, x):
            """
            Получение подвыборки по имени предиктора(ов)

            Входные данные:
            ---------------
                x : str/int/list-like
                        Предиктор или список предикторов

            Возвращает:
            -----------
                self : pd.DataFrame
                        Часть датафрейма (самого себя)
            """
            if isinstance(x, (list, set, tuple)):
                return self[self['predictor'].isin(x)]
            else:
                return self[self['predictor'] == x]

        def append(self, other):
            return super(GroupedPredictor, self).append(other)
a = pd.DataFrame({'a': [1,2,3], 'predictor':['a', 'a', 'b']})
# GroupedPredictor(a).append(a)
GroupedPredictor({'a': [1,2,3], 'predictor':['a', 'a', 'b']})
GroupedPredictor().append(a)


# %% ExecuteTime={"end_time": "2020-07-03T18:53:29.114158Z", "start_time": "2020-07-03T18:53:28.945370Z"} code_folding=[0, 1, 7]
# Проверка группировки функцией и классом
regul_grouping = (regul.grouped[['predictor', 'sample_count', 'target_count', 'value']]
                  .sort_values(['predictor', 'value'])
                  .reset_index(drop=True)
                  .replace({'nan':'NO_INFO'}))

test_grouping = pd.DataFrame()
for col in cols:
    tmp = cat_features_alpha_logloss(data, col, 'target', alphas, n_seeds)
    tmp.insert(0, 'predictor', col)
    test_grouping = test_grouping.append(tmp.rename(columns={'groups':'value'}))

test_grouping.sort_values(['predictor', 'value'], inplace=True)
test_grouping.reset_index(drop=True, inplace=True)
test_grouping.shape
(test_grouping == regul_grouping).sum().sum() - regul_grouping.shape[0] * regul_grouping.shape[1]

# %% ExecuteTime={"end_time": "2020-07-03T18:50:32.098173Z", "start_time": "2020-07-03T18:50:31.876664Z"} code_folding=[0, 1, 7] deletable=false editable=false run_control={"frozen": true}
# # Проверка расчета статистик функцией и классом
# regul_stats = (regul.stats[test_stats.columns]
#               .sort_values(['predictor', 'groups'])
#               .reset_index(drop=True)
#               .replace({'nan':'NO_INFO'}))
#
# test_stats = pd.DataFrame()
# for col in cols:
#     tmp = cat_features_alpha_logloss(data, col, 'mob12_90', alphas, n_seeds)
#     tmp.insert(0, 'predictor', col)
#     test_stats = test_stats.append(tmp)
#
# test_stats.sort_values(['predictor', 'groups'], inplace=True)
# test_stats.reset_index(drop=True, inplace=True)
# (test_stats.round(6) == regul_stats.round(6)).sum().sum() - regul_stats.shape[0] * regul_stats.shape[1]

# %% ExecuteTime={"end_time": "2020-07-03T19:07:39.853780Z", "start_time": "2020-07-03T19:06:58.385159Z"}
# Проверка расчета статистик функцией и классом
# regul_stats = (regul.stats[test_stats.columns]
#               .sort_values(['predictor', 'groups'])
#               .reset_index(drop=True)
#               .replace({'nan':'NO_INFO'}))
data = pd.concat([X, y], axis=1)
data.columns = ['col_'+str(i) for i in data.columns[:-1]] + ['target']
test_stats = pd.DataFrame()
for col in data.columns[:-1]:
    tmp = cat_features_alpha_logloss(data, col, 'target', alphas, n_seeds, plot_i=True)
    print(col, tmp)

# %% ExecuteTime={"end_time": "2020-07-03T19:07:39.853780Z", "start_time": "2020-07-03T19:06:58.385159Z"}
regul.alpha_values


# %% ExecuteTime={"end_time": "2020-07-03T19:00:48.162473Z", "start_time": "2020-07-03T19:00:48.156485Z"}
# функция WOE преобразования и расчета IV для датафрейма с готовой группировкой
def IVWOE(DF_groups):
    """
    Функция WOE преобразования и расчета IV для датафрейма с готовой группировкой
    Расчет статистики по группам
    DF_groups[['sample_count', 'target_count', 'groups']] - таблица данных по группам
    """
    nothing = 10 ** -6 # для случаев когда нет целевых событий для значения переменной
    DF_statistic = DF_groups
    DF_statistic['sample_rate'] = DF_statistic['sample_count'] / DF_statistic['sample_count'].sum()
    DF_statistic['target_rate'] = DF_statistic['target_count'] / DF_statistic['sample_count']

    # Расчет WoE и IV
    samples_num = DF_statistic['sample_count'].sum()
    events = DF_statistic['target_count'].sum()
    non_events = samples_num - events

    DF_statistic['non_events_i'] = DF_statistic['sample_count'] - DF_statistic['target_count']
    DF_statistic['event_rate_i'] = DF_statistic['target_count'] / events
    DF_statistic['non_event_rate_i'] = DF_statistic['non_events_i'] / non_events

    DF_statistic['WOE'] = [math.log(DF_statistic['non_event_rate_i'][i] / (DF_statistic['event_rate_i'][i] + nothing) + nothing) for i in DF_statistic.index]
    DF_statistic['IV'] = DF_statistic['WOE'] * (DF_statistic['non_event_rate_i'] - DF_statistic['event_rate_i'])

    return DF_statistic


# %% ExecuteTime={"end_time": "2020-07-03T19:01:48.438171Z", "start_time": "2020-07-03T19:01:48.409565Z"} code_folding=[]
# функция расчета IV, GINI и logloss для категориальных переменных с корректировкой целевой по alpha
from sklearn.metrics import log_loss


def cat_features_alpha_logloss(df, predictor, target, alpha, seed = 100, plot_i = False):
    """
    функция расчета IV, GINI и logloss для категориальных переменных с корректировкой целевой по alpha

    """

    L_logloss_mean = []
    GINI_IV_mean = []
    for alpha_i in alpha:
        logloss_i = []
        GINI_i = []
        IV_i = []
        for seed_i in range(seed):
            X_train, X_test, y_train, y_test = train_test_split(df[[predictor]], df[target],
                                                    test_size=0.3, random_state=seed_i, stratify=df[target])
            X_train = X_train.fillna('NO_INFO').astype(str)
            X_test = X_test.fillna('NO_INFO').astype(str)
            X_train[target] = y_train
            X_test[target] = y_test
            X_test = X_test[[predictor, target]]
            X_test_WOE = pd.DataFrame()
            X_test_WOE['Target'] = X_test[target]

            tmp = pd.crosstab(X_train[predictor], X_train[target], normalize='index')
            tmp.rename(columns={0:'Non Target', 1:'Target'}, inplace=True)
            tmp_values = pd.DataFrame({predictor: X_train[predictor].value_counts().index,
                                       'Values' : X_train[predictor].value_counts().values})
            tmp = pd.merge(tmp, tmp_values, how='left', on=predictor)
            tmp['Target_cnt'] = [int(x) for x in (tmp['Target'] * tmp['Values'])]

            # расчет оптимальной целевой для группы, формула и детали в видео
            # https://www.youtube.com/watch?v=g335THJxkto&list=PLLIunAIxCvT8ZYpC6-X7H0QfAQO9H0f-8&index=12&t=0s
            # pd = (y_local * K + Y_global * alpha) / (K + alpha)
            Y_global = y_train.mean()
            tmp['Target_transformed'] = ((tmp['Target']) * (tmp['Values'] / X_train.shape[0]) + Y_global * alpha_i) / ((tmp['Values'] / X_train.shape[0]) + alpha_i)
            tmp['Target_cnt_transformed'] = [math.floor(x) for x in tmp['Values'] * tmp['Target_transformed']]

            # если пустых значений = 1 - необходимо добавить в таблицу это значение
            if 'NO_INFO' not in tmp[predictor].values:
                tmp = tmp.append({predictor : 'NO_INFO',
                                'Non Target' : df[(df[predictor] == 'NO_INFO') & (df[target] == 0)].shape[0],
                                'Target' : df[(df[predictor] == 'NO_INFO') & (df[target] == 1)].shape[0],
                                'Values' : df[(df[predictor] == 'NO_INFO')].shape[0],
                                'Target_cnt' : df[(df[predictor] == 'NO_INFO') & (df[target] == 1)].shape[0],
                                'Target_transformed' : X_train[target].mean(),
                                'Target_cnt_transformed' : (df[(df[predictor] == 'NO_INFO')].shape[0]) * X_train[target].mean()
                               }, ignore_index=True)

            tmp.sort_values(by = 'Values', inplace=True, ascending=False)
            tmp = tmp.reset_index(drop=True)
            order = list(tmp[predictor])

            # расчет WOE и IV на Train
            df_i = tmp[['Values', 'Target_cnt_transformed', predictor]]
            df_i.rename(columns={'Values' : 'sample_count',
                                 'Target_cnt_transformed' : 'target_count',
                                  predictor : 'groups'}, inplace=True)
            WOE_i = IVWOE(df_i)

            # задаем промежуточную функцию для WOE преобразования переменной из исходного датафрейма по рассчитанным WOE из IVWOE
            def calc_woe_i(row_value):
                if row_value not in WOE_i['groups']:
                    return 0
                else:
                    i = 0
                    while row_value not in WOE_i['groups'][i]: i += 1
                    return WOE_i['WOE'][i]

            X_test_WOE['WOE'] = X_test[predictor].apply(calc_woe_i)
            roc_auc_i = roc_auc_score(X_test_WOE['Target'], X_test_WOE['WOE'])


            X_test = pd.merge(X_test, tmp[[predictor, 'Target_transformed']], how='left', on=predictor)
            #print(X_test[X_test['Target_transformed'].isna()])

#             print(seed_i)
#             print(X_test['Target_transformed'].isnull().sum())
#             print(X_test['Target_transformed'].loc[X_test['Target_transformed'].isnull()])
#             print(np.isinf(X_test['Target_transformed']).sum())

#             logloss_i.append(log_loss(X_test[target], X_test['Target_transformed']))
            logloss_i.append(log_loss(X_test[target], X_test['Target_transformed'].fillna(0)))
            IV_i.append(WOE_i['IV'].sum())
            GINI_i.append(abs(2 * roc_auc_i - 1))

        L_logloss_mean.append([alpha_i, np.mean(logloss_i)])
        GINI_IV_mean.append([alpha_i, np.mean(GINI_i), np.mean(IV_i)])

    df_cat_features_alpha_GINI_IV = pd.DataFrame(GINI_IV_mean, columns=['alpha', 'GINI', 'IV'])

    df_cat_features_alpha_logloss = pd.DataFrame(L_logloss_mean, columns=['alpha', 'logloss'])
    logloss_min = df_cat_features_alpha_logloss['logloss'].min()
    alpha_opt = df_cat_features_alpha_logloss[df_cat_features_alpha_logloss['logloss'] == logloss_min]['alpha'].values[0]

#     print('feature =', predictor)
#     print('log loss min =', logloss_min)
#     print('alpha optimum =', alpha_opt)

    if plot_i:
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111)
        ax.plot(df_cat_features_alpha_logloss['alpha'], df_cat_features_alpha_logloss['logloss'], label = 'logloss_test', marker='o', ms = 3, color = 'red')
        ax2 = ax.twinx()
        ax2.plot(df_cat_features_alpha_GINI_IV['alpha'], df_cat_features_alpha_GINI_IV['IV'], label = 'IV_train', marker='o', ms = 3, color = 'blue')
        ax2.plot(df_cat_features_alpha_GINI_IV['alpha'], df_cat_features_alpha_GINI_IV['GINI'], label = 'GINI_test', marker='o', ms = 3, color = 'green')

        ax_y_step = (max(df_cat_features_alpha_logloss['logloss']) - min(df_cat_features_alpha_logloss['logloss'])) * 0.1
        ax_y_min = min(df_cat_features_alpha_logloss['logloss']) - ax_y_step
        ax_y_max = max(df_cat_features_alpha_logloss['logloss']) + ax_y_step
        ax.set_ylim(ax_y_min, ax_y_max)

        ax2_y_step = (max(max(df_cat_features_alpha_GINI_IV['IV']), max(df_cat_features_alpha_GINI_IV['GINI'])) - min(min(df_cat_features_alpha_GINI_IV['IV']), min(df_cat_features_alpha_GINI_IV['GINI']))) * 0.1
        ax2_y_min = min(min(df_cat_features_alpha_GINI_IV['IV']), min(df_cat_features_alpha_GINI_IV['GINI'])) - ax2_y_step
        ax2_y_max = max(max(df_cat_features_alpha_GINI_IV['IV']), max(df_cat_features_alpha_GINI_IV['GINI'])) + ax2_y_step
        ax2.set_ylim(ax2_y_min, ax2_y_max)

        ax.tick_params(axis="x", labelsize=12)
        ax2.tick_params(axis="x", labelsize=12)
        ax.set_xlabel('alpha', fontsize=16)
        ax.set_ylabel('logloss', fontsize=16)
        ax2.set_ylabel('GINI and IV', fontsize=16)
        ax.legend(loc = "upper left")
        ax2.legend(loc = "upper right")
        plt.grid(True)
        plt.title('Распределение logloss, GINI и IV от значения alpha', fontsize=20)
        plt.show()

    return(alpha_opt)

# %%
