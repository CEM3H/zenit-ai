# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: all,-language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.2
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

# %% ExecuteTime={"end_time": "2020-04-28T06:15:50.988629Z", "start_time": "2020-04-28T06:15:49.240979Z"}
import math
import time

import sys

import woeTransformer as woe_vanilla
import woeTransformer_beta as woe_new
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ## Данные для теста

# %% ExecuteTime={"end_time": "2020-04-28T06:15:51.022760Z", "start_time": "2020-04-28T06:15:50.989632Z"} code_folding=[]
np.random.seed(42)
# Датасет для обучения
df_0 = pd.DataFrame({'digits':np.random.choice(range(10), 10000),
                     'integers':np.random.choice(range(100), 10000),
                     'floats': np.random.choice(np.linspace(1, 10, 100), 10000),
                     'integers_w_neg':np.random.choice(range(-50, 51), 10000),
                     'integers_w_small_cat':np.hstack((np.array(np.random.choice(range(10), 9990), float),
                                                   [100]*10)),
                     'floats_w_na':np.hstack((np.array(np.random.choice(range(10), 9000), float),
                                                     np.full(1000, np.nan))),
                     'floats_w_letters':np.hstack((np.random.choice(range(10), 9000),
                                                 np.array(['d', 'f']*500))),
                     'integers_w_letters_obj':np.hstack((np.array(np.random.choice(range(10), 9000), int),
                                                     np.array(['d', 'f']*500, object))),
                     'letters':np.random.choice(['a', 'b', 'c', 'd', 'e', 'f', 'g'], 10000),
                     'letters_uneq_freq': np.random.choice(['a', 'b', 'c'], 10000, p=[0.6, 0.3, 0.1]),
                     'letters_w_na':np.hstack((np.array(np.random.choice(['a', 'b', 'c'], 9000), object),
                                               np.full(1000, np.nan))),
                     'single_integer': np.ones(10000),
                     'single_letter': np.array(['x']*10000),
                     'single_nan': np.full(10000, np.nan),
                     'target':np.random.choice(range(2), 10000)})

# Датасет для применения группировки
df_0_test = pd.DataFrame({'digits':np.random.choice(range(10), 10000,
                                                    p=[.15,.15,.05,.05, .1,.1,.1,.01,.19,.1]),   #изменены веса
                         'floats': np.random.choice(np.linspace(1, 20, 100), 10000),
                         'integers':np.random.choice(range(100), 10000),
                         'integers_w_neg':np.random.choice(range(-100, 101), 10000),                 #изменен диапазон значений
                         'floats_w_small_cat':np.hstack((np.array(np.random.choice(range(10), 9990), float),
                                                       [100]*10)),
                         'floats_w_na':np.hstack((np.array(np.random.choice(range(10), 8000), float),
                                                         np.full(2000, np.nan))),                # увеличено кол-во пустышек
                         'integers_w_letters':np.hstack((np.random.choice(range(10), 9000),
                                                     np.array(['d', 'X']*500))),                 # одна категория заменена
                         'integers_w_letters_obj':np.hstack((np.array(np.random.choice(range(10), 9000), int),
                                                         np.array(['d', 'f', 'y', 'z']*250, object))),   # добавлены 2 новых 
                         'letters':np.random.choice(['a', 'b', 'c', 'd', 'e', 'f', 'g'], 10000),
                         'letters_uneq_freq': np.random.choice(['a', 'b', 'c'], 10000, p=[0.1, 0.6, 0.3]),   #изменены веса
                         'letters_w_na':np.hstack((np.array(np.random.choice(['a', 'b', 'c', 'd'], 9000), object),
                                                   np.full(1000, np.nan))), #добавлена категория
                         'letters_crazy':['XYZ']*10000,
                         'single_num': np.ones(10000),
                         'single_str': np.array(['x']*10000),
                         'single_nan': np.full(10000, np.nan),
                         'target':np.random.choice(range(2), 10000)})


# %% [markdown]
# ## Класс

# %% [markdown]
# ### Особенности класса

# %% [markdown]
# * поддерживает методы `fit` / `transform`/ `fit_transform`
# * позволяет отрисовывать графики по одной, нескольким или по всем преобразованным переменным
# * обрабатывает пропущенные значения как отдельную категорию
# * хранит итоговую таблицу с результатами групировки всех предикторов
# * хранит дополнительную информацию о преобразовании каждого предиктора
#     * список сгруппированных предикторов
#     * указанные при обучении категориальные значения 
#     * значения, которые могут являться категориями
#     * группы, не удовлетворяющие заданным условиям
#     * монотонные границы (для тех признаков, по которым удалось построить границы)
#     * коэффициенты тренда (для тех признаков, по которым удалось построить границы)

# %% [markdown]
# ### Сам класс
#

# %% ExecuteTime={"end_time": "2020-04-28T06:15:51.081879Z", "start_time": "2020-04-28T06:15:51.023725Z"} code_folding=[41, 109, 135, 162, 187, 252, 336, 345, 419, 458, 506, 522]
class WoeTransformer:
    """
    Класс для построения и применения WOE группировки к датасету
    
    Параметры:
    ----------
        min_sample_rate : float, default 0.05
                Минимальный размер группы (доля от размера выборки)
        min_count : int, default 3
                Минимальное количество наблюдений каждого класса в группе
#        detect_categories
    Атрибуты:
        stats : pd.DataFrame
                Результаты WOE-группировки по всем предикторам
        grouped : pd.DataFrame
                Результаты агрегации предикторов
        cat_values : dict
                Спискок категорий по предикторам: переданный при обучении
        trend_coefs : dict of np.arrays
                Коэффициенты тренда по количественным предикторам
        borders : dict of lists
                Монотонные границы по количественным предикторам
        possible groups : pd.DataFrame
                Данные о значениях предиктора, которые могли бы стать 
                отдельными категориями        
        bad_groups : pd.DataFrame
                Данные о группах, которые не удовлетворяют условиям
    
    Методы:
    -------
        fit
                Обучение трансформера, сохранение всех нужных данных
        fit_transform
                Обучение трансформера и применение группировки к данным
        transform
                Применение обученного трансформера к новым данным
        plot_woe
            
            
    """
    
    class _GroupedPredictor(pd.DataFrame):
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
    
    def __init__(self, min_sample_rate=0.05, min_count=3):
        """
        Инициализация экземпляра класса
        
        Параметры:
        ----------
            min_sample_rate : float, default 0.05
                    Минимальный размер группы (доля от размера выборки)
            min_count : int, default 3
                    Минимальное количество наблюдений каждого класса в группе
        """
        self.min_sample_rate = min_sample_rate
        self.min_count = min_count
        self.predictors = []
#         self.detect_categories = detect_categories   
    
    def fit(self, X, y, cat_values={}):
        """
        Обучение трансформера и расчет всех промежуточных данных
        
        Входные данные:
        ---------------
            X : pd.DataFrame
                    Датафрейм с предикторами, которые нужно сгруппировать
            y : pd.Series
                    Целевая переменная
            cat_values : dict, optional
                    Словарь списков с особыми значениями, которые нужно 
                    выделить в категории
                    По умолчанию все строковые и пропущенные значения 
                    выделяются в отдельные категории
        """
        # Сброс текущего состояния трансформера
        self._reset_state()
        self.cat_values = cat_values
        # Агрегация значений предикторов
        self._grouping(X, y)
        # Расчет WOE и IV
        self._fit_numeric(X, y)
        # Поиск потенциальных групп
        self._get_possible_groups()
        # Поиск "плохих" групп
        self._get_bad_groups()
   
    
    def plot_woe(self, predictors=None):
        """
        Отрисовка одного или нескольких графиков группировки
        
        Входные данные:
        ---------------
            predictors : str/list-like, default None
                    Предиктор(ы), по которым нужны графики
                    -- если str - отрисовывается один график
                    -- если list-like - отрисовываются графики из списка
                    -- если None - отрисовываются все сгруппированные предикторы 
        """
        if predictors is None:
            predictors = self.predictors
        elif isinstance(predictors, str):
            predictors = [predictors]
        elif isinstance(predictors, (list, tuple, set)):
            predictors = predictors
            
        _, axes = plt.subplots(figsize=(10, len(predictors)*5), nrows=len(predictors))
        try:
            for i, col in enumerate(predictors):
                self._plot_single_woe_grouping(self.stats.get_predictor(col), axes[i])
        except TypeError:
            self._plot_single_woe_grouping(self.stats.get_predictor(col), axes)
              
    def transform(self, X):
        """
        Применение обученного трансформера к новым данным
        
        Входные данные:
        ---------------
            X : pd.DataFrame
                    Датафрейм, который нужно преобразовать
                    Предикторы, которые не были сгруппированы ранее, будут
                    проигнорированы и выведется сообщение
        Возвращает:
        -----------
            transformed : pd.DataFrame
                    Преобразованный датасет
        """
        transformed = pd.DataFrame()
        if isinstance(X, pd.DataFrame):
            for i in X:
                if i in self.predictors:
                    transformed[i] = self._transform_single(X[i])
                else:
                    print(f"Column not in fitted predictors list: {i}") 
        elif isinstance(X, pd.Series):
            transformed = self._transform_single(X)
            
        return transformed
    
    def fit_transform(self, X, y, cat_values={}):
        """
        Обучение трансформера и расчет всех промежуточных данных 
        с последующим примененим группировки к тем же данным
        
        Входные данные:
        ---------------
            X : pd.DataFrame
                    Датафрейм с предикторами, которые нужно сгруппировать
            y : pd.Series
                    Целевая переменная
            cat_values : dict, optional
                    Словарь списков с особыми значениями, которые нужно 
                    выделить в категории
                    По умолчанию все строковые и пропущенные значения 
                    выделяются в отдельные категории
        Возвращает:
        -----------
            transformed : pd.DataFrame
                    Преобразованный датасет
        """
        
        self.fit(X, y, cat_values=cat_values)
        return self.transform(X)
    
    def _transform_single(self, S_data):
        """
        Применение группировки и WoE-преобразования

        Входные данные:
        ---------------
            S_data : pandas.Series
                    Значения предиктора
        Возвращает:
        ---------------
            X_woe : pandas.DataFrame 
                    WoE-преобразования значений предиктора
                    WoE = 0, если группа не встречалась в обучающей выборке

        """
        X_woe = S_data.copy()
        DF_groups = self.stats.get_predictor(X_woe.name)
        # Маппинги для замены групп на соответствующие значения WOE
        num_map = {DF_groups.loc[i, 'groups']: DF_groups.loc[i, 'WOE']
                   for i in DF_groups.index if DF_groups.loc[i, 'type'] == 'num'}
        cat_map = {DF_groups.loc[i, 'groups']: DF_groups.loc[i, 'WOE']
                   for i in DF_groups.index if DF_groups.loc[i, 'type'] == 'cat'}
        # Категориальные группы
        cat_bounds = DF_groups.loc[DF_groups['type'] == 'cat', 'groups']

        # predict по числовым значениям
        DF_num = DF_groups.loc[DF_groups['type'] == 'num']
        if DF_num.shape[0] > 0:
            # Границы (правые) интервалов для разбивки числовых переменных 
            num_bounds = [-np.inf] + list(pd.IntervalIndex(DF_groups.loc[DF_groups['type'] == 'num', 'groups']).right)
            # Выделение только числовых значений предиктора 
            # (похожих на числа и тех, что явно не указаны как категориальные)
            X_woe_num = pd.to_numeric(X_woe[(self._get_nums_mask(X_woe)) &
                                              (~X_woe.isin(cat_bounds))])
            # Разбивка значений на интервалы в соответствии с группировкой
            X_woe_num = pd.cut(X_woe_num, num_bounds)
            # Замена групп на значения WOE 
            X_woe_num = X_woe_num.replace(num_map)
            X_woe_num.name = 'woe'
        else:
            X_woe_num = pd.Series()

        # predict по категориальным значениям (может обновлять значения по числовым)
        DF_cat = DF_groups.loc[DF_groups['type'] == 'cat']
        if DF_cat.shape[0] > 0:
            # Выделение строковых значений и тех, что явно выделены как категориальные 
            X_woe_cat = X_woe[X_woe.isin(cat_map.keys())]
            # Замена групп на значения WOE 
            X_woe_cat = X_woe_cat.replace(cat_map)
        else:
            X_woe_cat = pd.Series()

        # predict по новым категориям (нечисловые: которых не было при групприровке)
        # Сбор индексов категориальных и числовых значений
        used_index = np.hstack([X_woe_cat.index, X_woe_num.index])
        if len(used_index) < len(S_data):
            X_woe_oth = X_woe.index.drop(used_index)
            X_woe_oth = pd.Series(0, index=X_woe_oth)
        else:
            X_woe_oth = pd.Series()

        X_woe = pd.concat([X_woe_num, X_woe_cat, X_woe_oth]).sort_index()

        return X_woe
    
    def _fit_numeric(self, X, y):
        """
        Расчет WOE и IV
        
        Входные данные:
        ---------------
            X : pd.DataFrame
                    Датафрейм с предикторами, которые нужно сгруппировать
            y : pd.Series
                    Целевая переменная
        """
        self.trend_coefs = {}
        self.borders = {}
        res = pd.DataFrame()
        for i in X:
            cat_vals = self.cat_values.get(i, [])
            nan_mask = X[i].isna()
            num_mask = self._get_nums_mask(X[i]) & (~X[i].isin(cat_vals)) & (~nan_mask)
            num_vals = X.loc[num_mask, i].unique()
            gr_subset = (self.grouped.get_predictor(i))

            # Расчет коэффициентов тренда по числовым значениям предиктора
            if num_mask.sum() > 0:
                self.trend_coefs.update({i: np.polyfit(X.loc[num_mask, i].astype(float), 
                                                       y.loc[num_mask], 
                                                       deg=1)})
                # Расчет монотонных границ
                gr_subset_num = gr_subset[gr_subset['value'].isin(num_vals)].copy()
                gr_subset_num['value'] = pd.to_numeric(gr_subset_num['value'])
                borders = self._monotonic_borders(gr_subset_num, 
                                                  self.trend_coefs[i])
                self.borders.update({i:borders})
                # Применение границ к сгруппированным данным
                gr_subset_num['groups'] = pd.cut(gr_subset_num['value'], borders)
                gr_subset_num['type'] = 'num'
            else:
                gr_subset_num = pd.DataFrame()
            
            
            # Расчет коэффициентов тренда по категориальным значениям предиктора
            if (~num_mask).sum() > 0:
                gr_subset_cat = gr_subset[~gr_subset['value'].isin(num_vals)].copy()
                gr_subset_cat['groups'] = gr_subset_cat['value'].fillna('пусто')
                gr_subset_cat['type'] = 'cat'
            else:
                gr_subset_cat = pd.DataFrame()
            
            # Объединение числовых и категориальных значений
            gr_subset = pd.concat([gr_subset_num, gr_subset_cat], axis=0, ignore_index=True)

            res_i = self._statistic(gr_subset)
            res_i['groups'].replace({'пусто':np.nan}, inplace=True) 
            
            res = res.append(res_i)
            self.predictors.append(i)
            
        self.stats = res
        self.stats = self._GroupedPredictor(self.stats)
        
    def _get_possible_groups(self):
        """
        Поиск возможных групп в значениях предикторов после агрегации
        """
        self.possible_groups = pd.DataFrame()
        # Выделение значений предиктора с достаточным кол-вом наблюдений и 
        # не отмеченных, как категориальные
        for i in self.predictors:
            cat_vals = self.cat_values.get(i, [])
            DF_i1 = self.grouped.get_predictor(i).copy()
            DF_i1 = DF_i1.loc[(DF_i1['sample_rate'] > self.min_sample_rate)
                              & (~DF_i1['value'].isin(cat_vals))
                             ]

            ## Выделение всех значений предиктора, не отмеченных, как категориальные
            DF_i2 = self.grouped.get_predictor(i).copy()
            DF_i2 = DF_i2.loc[(~DF_i2['value'].isin(cat_vals))]

            ## Выбор значений: которые не равны бесконености и при этом не являются числами
            L = (~(DF_i2['value'] == np.inf) & (~(self._get_nums_mask(DF_i2['value']))))
            DF_i2 = DF_i2.loc[L]
            # Объединение найденных значений в одну таблицу
            DF_i = pd.concat((DF_i1, DF_i2), ignore_index=True).drop_duplicates()
            
            self.possible_groups = self.possible_groups.append(DF_i)
              
    def _get_bad_groups(self):
        """
        Поиск групп: не удовлетворяющих условиям
        """     
        self.bad_groups = self.stats.loc[(self.stats['sample_rate'] < self.min_sample_rate)
                                | (self.stats['target_count'] < self.min_count)
                                | (self.stats['sample_count']
                                - self.stats['target_count'] < self.min_count)]
    
    def _monotonic_borders(self, DF_grouping, p):
        """
        Определение оптимальных границ групп предиктора (монотонный тренд)

        Входные данные:
        ---------------
            DF_grouping : pandas.DataFrame
                    Агрегированные данные по значениям предиктора (результат работы
                    фунции grouping, очищенный от категориальных значений).
                    Должен содержать поля 'predictor', 'sample_count', 'target_count', 
                    'sample_rate и 'target_rate'
            p : list-like, длиной в 2 элемента
                    Коэффициенты линейного тренда значений предиктора

        Возвращает:
        ---------------    
            R_borders : list
                 Правые границы групп для последующей группировки

        """
        k01, k11 = (1, 1) if p[0] > 0 else (0, -1)
        R_borders = []
        min_ind = 0  # минимальный индекс. Начальные условия

        while min_ind < DF_grouping.shape[0]:  # цикл по новым группам
            pd_gr_i = k01  # средняя pd в группе. Начальные условия (зависит от общего тренда)

            # Расчет показателей накопительным итогом
            DF_j = DF_grouping.loc[min_ind:]
            DF_iter = DF_j[['sample_rate', 'sample_count', 'target_count']].cumsum()
            DF_iter['non_target_count'] = DF_iter['sample_count'] - DF_iter['target_count']
            DF_iter['target_rate'] = DF_iter['target_count'] / DF_iter['sample_count']

            # Проверка на соответствие критериям групп
            DF_iter['check'] = ((DF_iter['sample_rate'] >= self.min_sample_rate - 10 ** -9)
                                & (DF_iter['target_count'] >= self.min_count)
                                & (DF_iter['non_target_count'] >= self.min_count))

            # Расчет базы для проверки оптимальности границы
            # В зависимости от тренда считается скользящий _вперед_ минимум или максимум 
            # (в расчете участвуют все наблюдения от текущего до последнего)
            if k11 == 1:
                DF_iter['pd_gr'] = DF_iter['target_rate'][::-1].rolling(len(DF_iter), min_periods=0).min()[::-1]
            else:
                DF_iter['pd_gr'] = DF_iter['target_rate'][::-1].rolling(len(DF_iter), min_periods=0).max()[::-1]

            # Проверка оптимальности границы
            DF_iter['opt'] = DF_iter['target_rate'] == DF_iter['pd_gr']
            DF_iter = pd.concat([DF_j[['value']], DF_iter], axis=1)
            try:
                min_ind = DF_iter.loc[(DF_iter['check'] == True)
                                      & (DF_iter['opt'] == True)
                                    , 'target_rate'].index.values[0]
                pd_gr_i = DF_iter.loc[min_ind, 'target_rate']
                score_j = DF_iter.loc[min_ind, 'value']
                if len(R_borders) > 0 and score_j == R_borders[-1]:  # Выход из цикла, если нет оптимальных границ
                    break
            except:
                break
            min_ind += 1
            R_borders.append(score_j)

        # Проверка последней добавленной группы
        DF_iter = DF_grouping.loc[DF_grouping['value'] > R_borders[-1]]
        sample_rate_i = DF_iter['sample_rate'].sum()  # доля выборки
        sample_count_i = DF_iter['sample_count'].sum()  # количество наблюдений
        target_count_i = DF_iter['target_count'].sum()  # количество целевых
        non_target_count_i = sample_count_i - target_count_i  # количество нецелевых

        if (sample_rate_i < self.min_sample_rate) or (target_count_i < self.min_count) or (non_target_count_i < self.min_count):
            R_borders.remove(R_borders[-1])  # удаление последней границы
        R_borders = [-np.inf] + R_borders + [np.inf]
        return R_borders

    def _grouping(self, X, y):
        """
        Агрегация данных по значениям предиктора. 
        Рассчитывает количество наблюдений,
        количество целевых событий, долю группы от общего числа наблюдений 
        и долю целевых в группе 

        Входные данные:
        ---------------
            X : pandas.DataFrame
                    Таблица данных для агрегации, должна содержать поля 'predictor' и 'target'. 
                    Поле target при этом должно состоять из 0 и 1, где 1 - целевое событие
            y : pandas.Series
                    Целевая переменная

        """
                 
        df = X.copy()
        df.fillna('пусто', inplace=True)
        df['target'] = y.copy()
        self.grouped = pd.DataFrame()

        # Группировка и расчет показателей
        for col in df.columns[:-1]:
            grouped_temp = df.groupby(col)['target'].agg(['count', 'sum']).reset_index()
            grouped_temp.columns = ['value', 'sample_count', 'target_count']
            grouped_temp['sample_rate'] = grouped_temp['sample_count'] / grouped_temp['sample_count'].sum()
            grouped_temp['target_rate'] = grouped_temp['target_count'] / grouped_temp['sample_count']
            grouped_temp.insert(0, 'predictor', col)
            self.grouped = self.grouped.append(grouped_temp)
        
        # Замена пустых значений обратно на np.nan ИЛИ преобразование в числовой тип
        try:
            self.grouped['value'] = self.grouped['value'].replace({'пусто':np.nan})
        except TypeError:
            self.grouped['value'] = pd.to_numeric(self.grouped['value'], downcast='signed')

        self.grouped = self._GroupedPredictor(self.grouped)
    
    def _statistic(self, DF_groups):
        """
        Расчет статистики по группам предиктора: минимальное, максимальное значение, доля от 
        общего объема выборки, количество и доля целевых и нецелевых событий в каждой группе
        А также расчет WOE и IV каждой группы

        Входные данные:
        ---------------
            DF_groups : pandas.DataFrame
                    Данные полученных групп предиктора. Кол-во строк совпадает с кол-вом
                    уникальных значений предиктора. 
                    Должен содержать столбцы: 'sample_count', 'target_count', 'groups'
        Возвращает:
        ---------------
            stats : pandas.DataFrame
                    Агрегированные данные по каждой группе

        """
        nothing = 10 ** -6
        stats = DF_groups.groupby(['predictor', 'groups'],sort=False).agg({'type':'first',
                                                 'sample_count':'sum', 
                                                 'target_count':'sum', 
                                                 'value':['min', 'max']},
#                                                 as_index=False, 
                                                 
                                                              )
        stats.columns = ['type', 'sample_count', 'target_count', 'min', 'max']
        stats.reset_index(inplace=True)
        stats['sample_rate'] = stats['sample_count'] / stats['sample_count'].sum()
        stats['target_rate'] = stats['target_count'] / stats['sample_count']

        # Расчет WoE и IV
        samples_num = stats['sample_count'].sum()
        events = stats['target_count'].sum()
        non_events = samples_num - events

        stats['non_events_i'] = stats['sample_count'] - stats['target_count']
        stats['event_rate_i'] = stats['target_count'] / (events + nothing)
        stats['non_event_rate_i'] = stats['non_events_i'] / (non_events + nothing)

        stats['WOE'] = np.log(stats['non_event_rate_i']
                                     / (stats['event_rate_i'] + nothing)
                                     + nothing)

        stats['IV'] = stats['WOE'] * (stats['non_event_rate_i'] - stats['event_rate_i'])

        return stats
    
    def _calc_trend_coefs(self, x, y):
        """
        Расчет коэффициентов тренда
        
        Входные данные:
        ---------------
                x : pandas.Series
                        Значения предиктора
                y : pandas.Series
                        Целевая переменная
        Возвращает:
        -----------
                tuple - коэффициенты
        """
        return {x.name : np.polyfit(x, y, deg=1)}
    
    def _plot_single_woe_grouping(self, stats, ax_pd=None):
        """
        Построение графика по группировке предиктора

        Входные данные:
        ---------------
            stats : pandas.DataFrame
                    Статистика по каждой группе (результат работы функции statistic):
                        минимальное, максимальное значение, доля от общего объема выборки,
                        количество и доля целевых и нецелевых событий в каждой группе,
                        WOE и IV каждой группы
                    Должен содержать столбцы: 'sample_rate', 'target_rate', 'WOE'

        """
        ## Расчеты
        x2 = [stats['sample_rate'][:i].sum()
              for i in range(stats.shape[0])] + [1]  # доля выборки с накоплением
        x = [np.mean(x2[i:i + 2]) for i in range(len(x2) - 1)]  # средняя точка в группах

        # Выделение нужной информации для компактности
        woe = list(stats['WOE'])
        height = list(stats['target_rate'])  # проблемность в группе
        width = list(stats['sample_rate'])  # доля выборки на группу

        ## Визуализация
        if ax_pd is None:
            _, ax_pd = plt.subplots(figsize=(8, 5))

        # Столбчатая диаграмма доли целевых в группах
        bar_pd = ax_pd.bar(x=x, height=height, width=width,
                           color=[0, 122 / 255, 123 / 255], label='Группировка',
                           alpha=0.7)

        # График значений WOE по группам
        ax_woe = ax_pd.twinx()  # дубликат осей координат
        line_woe = ax_woe.plot(x, woe, lw=2,
                               color=[37 / 255, 40 / 255, 43 / 255],
                               label='woe', marker='o')

        # Линия нулевого значения WOE
        line_0 = ax_woe.plot([0, 1], [0, 0], lw=1,
                             color=[37 / 255, 40 / 255, 43 / 255], linestyle='--')

        # Настройка осей координат
        plt.xlim([0, 1])
        plt.xticks(x2, [round(i, 2) for i in x2], fontsize=12)
        ax_pd.grid(True)
        ax_pd.set_xlabel('Доля выборки', fontsize=16)
        ax_pd.set_ylabel('pd', fontsize=16)
        ax_woe.set_ylabel('woe', fontsize=16)

        ## Расчет границ графика и шага сетки
        max_woe = max([int(abs(i)) + 1 for i in woe])
        max_pd = max([int(i * 10) + 1 for i in height]) / 10
        
        # Границы и сетка для столбчатой диаграммы
        ax_pd.set_ylim([0, max_pd])
        ax_pd.set_yticks([round(i, 2) for i in np.linspace(0, max_pd, 11)])
        ax_pd.legend(bbox_to_anchor=(1.05, .83), loc=[0.2, -0.25], fontsize=14)

        # Границы и сетка для графика WOE
        ax_woe.set_ylim([-max_woe, max_woe])
        ax_woe.set_yticks([round(i, 2) for i in np.linspace(-max_woe, max_woe, 11)])
        ax_woe.legend(bbox_to_anchor=(1.05, .92), loc=[0.6, -0.25], fontsize=14)
        

        plt.title(f'Группировка предиктора {stats.loc[0, "predictor"]}', fontsize=18)

        # Для категориальных
        n_cat = stats.loc[stats['type'] == 'cat'].shape[0]

        if n_cat > 0:
            bar_pd = ax_pd.bar(x=x[-n_cat:], height=height[-n_cat:], width=width[-n_cat:], color='m',
                               label='Категориальные')
            ax_pd.legend(bbox_to_anchor=(1.05, 0.76), loc=[0.15, -0.33], fontsize=14)
        
        plt.tight_layout()
   
    # Служебные функции
    def _reset_state(self):
        self.cat_values = {}
        self.predictors = []
        
    def _get_nums_mask(self, x):
        mask = pd.to_numeric(x, errors='coerce').notna()
        return mask
    

# %% [markdown]
# ## Инициализация и обучение группировки на датасете

# %% ExecuteTime={"end_time": "2020-04-28T06:15:51.092942Z", "start_time": "2020-04-28T06:15:51.082882Z"} code_folding=[] scrolled=false
# Инициализация класса и вывод начальных параметров
woe = WoeTransformer()
print('Min sample rate:', woe.min_sample_rate)
print('Min count:', woe.min_count)
print('Fitted predictors:', woe.predictors)

# %% ExecuteTime={"end_time": "2020-04-28T06:15:51.801704Z", "start_time": "2020-04-28T06:15:51.094415Z"}
# Обучение трансформера - получение группировки и сопутсвующей информации 
woe.fit(df_0.iloc[:, :-1], df_0['target'], cat_values={'nums_w_small_cat':[100]})

# %% [markdown]
# ## Результаты группировки

# %% ExecuteTime={"end_time": "2020-04-28T06:15:51.807721Z", "start_time": "2020-04-28T06:15:51.802707Z"}
print('Сгруппированные предикторы:\n')
woe.predictors

# %% ExecuteTime={"end_time": "2020-04-28T06:15:51.817245Z", "start_time": "2020-04-28T06:15:51.809224Z"}
print('Категориальные значения предикторов:\n')
woe.cat_values

# %% ExecuteTime={"end_time": "2020-04-28T06:15:51.824265Z", "start_time": "2020-04-28T06:15:51.819251Z"}
print('Коэффиценты тренда числовых переменных:')
woe.trend_coefs

# %% ExecuteTime={"end_time": "2020-04-28T06:15:51.831285Z", "start_time": "2020-04-28T06:15:51.826271Z"}
print('Монотонные границы:')
woe.borders

# %% ExecuteTime={"end_time": "2020-04-28T06:15:51.851338Z", "start_time": "2020-04-28T06:15:51.833291Z"} scrolled=true
print('Возможные категории:')
woe.possible_groups

# %% ExecuteTime={"end_time": "2020-04-28T06:15:51.861364Z", "start_time": "2020-04-28T06:15:51.852340Z"}
print('Категории: не удовлетворяющие условиям:')
woe.bad_groups

# %% ExecuteTime={"end_time": "2020-04-28T06:15:51.872394Z", "start_time": "2020-04-28T06:15:51.862366Z"}
print('Агрегированные значения предикторов:')
print('Размерность:', woe.grouped.shape)
woe.grouped.head()

# %% ExecuteTime={"end_time": "2020-04-28T06:15:51.894452Z", "start_time": "2020-04-28T06:15:51.875404Z"}
print('Результаты WOE-группировки:')
woe.stats

# %% ExecuteTime={"end_time": "2020-04-28T06:15:52.100950Z", "start_time": "2020-04-28T06:15:51.895454Z"}
print('График отдельного предиктора')
woe.plot_woe('letters_uneq_freq')

# %% ExecuteTime={"end_time": "2020-04-28T06:15:54.893306Z", "start_time": "2020-04-28T06:15:52.102130Z"} scrolled=true
print('Все графики')
woe.plot_woe()

# %% [markdown]
# ## Обучение с указанием категориальных переменных

# %% [markdown]
# Особые категории передаются с помощью словаря списков

# %% ExecuteTime={"end_time": "2020-04-28T06:15:54.904363Z", "start_time": "2020-04-28T06:15:54.894344Z"}
df_0.head()

# %% ExecuteTime={"end_time": "2020-04-28T06:15:55.535248Z", "start_time": "2020-04-28T06:15:54.906343Z"}
# Обучение трансформера - получение группировки и сопутсвующей информации 
woe = WoeTransformer()
try: 
    woe.cat_values
except AttributeError: 
    print('Категориальные значения не определены')
woe.fit(df_0.iloc[:, :-1], df_0['target'], cat_values={'nums_w_small_cat':[100],
                                                       'nums_w_neg':[-1, -11],
                                                       'nums_w_na':[1,3,5],
                                                       'nums_w_letters':[100],  # такого значеня в предиторке нет
                                                       },)
woe.cat_values

# %% ExecuteTime={"end_time": "2020-04-28T06:15:55.550914Z", "start_time": "2020-04-28T06:15:55.536376Z"} scrolled=true
# Результаты группировки по предикторам с категориями
woe.stats[woe.stats.predictor.isin(woe.cat_values.keys())]

# %% ExecuteTime={"end_time": "2020-04-28T06:15:55.574092Z", "start_time": "2020-04-28T06:15:55.552418Z"}
# Потенциально "плохие" категории
woe.bad_groups

# %% [markdown]
# ## Применение группировки к тем же данным

# %% ExecuteTime={"end_time": "2020-04-28T06:15:56.764165Z", "start_time": "2020-04-28T06:15:55.575480Z"} scrolled=false
woe = WoeTransformer()
woe.fit(df_0.iloc[:,:-1], df_0['target'])
transformed_same = woe.transform(df_0.iloc[:,:-1])
transformed_same

# %% ExecuteTime={"end_time": "2020-04-28T06:15:56.781678Z", "start_time": "2020-04-28T06:15:56.765635Z"} code_folding=[] scrolled=true
# Количество групп и кол-во значений в преобразованном предикторе
for col in df_0.iloc[:,:-1]:
    print(f'{col}:', len(woe.stats.get_predictor(col)), transformed_same[col].nunique())

# %% [markdown]
# ## Применение группировки к новым данным

# %% ExecuteTime={"end_time": "2020-04-28T06:15:58.611963Z", "start_time": "2020-04-28T06:15:57.407348Z"}
transformed_other = woe.transform(df_0_test.iloc[:,:-1])
transformed_other

# %% ExecuteTime={"end_time": "2020-04-28T06:15:58.630344Z", "start_time": "2020-04-28T06:15:58.612965Z"} code_folding=[] scrolled=false
# Количество групп и кол-во значений в преобразованном предикторе
print("Cписок значений был изменен в столбцах:")
print(' nums_w_letters\n', 'nums_w_letters_obj\n', 'letters_w_na')
print()
for col in df_0_test.iloc[:,:-1]:
    if col in transformed_other.columns:
        a = woe.stats.get_predictor(col).WOE
        b = transformed_other[col]
        print(f'{col}:', a.nunique(), b.nunique(), 'Новые:', b[~b.isin(a.unique())].unique())

# %% [markdown]
# ## Одновременное обучение и применение группировки

# %% ExecuteTime={"end_time": "2020-04-28T06:15:58.611963Z", "start_time": "2020-04-28T06:15:57.407348Z"}
fit_transformed_same = woe.fit_transform(df_0.iloc[:,:-1], df_0['target'])
fit_transformed_same

# %% ExecuteTime={"end_time": "2020-04-28T06:15:58.630344Z", "start_time": "2020-04-28T06:15:58.612965Z"} code_folding=[] scrolled=true
# Количество групп и кол-во значений в преобразованном предикторе
for col in df_0.iloc[:,:-1]:
    print(f'{col}:', len(woe.stats.get_predictor(col)), transformed_same[col].nunique())

# %% [markdown]
# ## Сравнение группировки с классическим трансформером

# %% ExecuteTime={"end_time": "2020-04-28T06:16:08.455912Z", "start_time": "2020-04-28T06:15:58.631318Z"}
# Обучение и применение группировок классическим вариантом
res_df = pd.DataFrame()
res_df_trained = pd.DataFrame()
for col in df_0.columns[:-1]:
    print(col)
    cv = ['пусто']
    if 'nums_w_letters' in col:
        cv.extend(['d', 'f'])
    elif 'letters' in col:
        cv.extend(df_0[col].unique())

    t = woe_vanilla.woeTransformer(df_0[col].fillna('пусто'), df_0['target'], plot=True, cat_values=cv)
    o = woe_vanilla.woe_apply(df_0[col].fillna('пусто'), t)
    o.name = col
    t.insert(1, 'predictor', col)
    res_df = res_df.append(t)
    res_df_trained = res_df_trained.append(o)

# %% ExecuteTime={"end_time": "2020-04-28T06:16:08.472425Z", "start_time": "2020-04-28T06:16:08.457283Z"} scrolled=true
# результаты группировки классом
res_class = woe.stats.copy()
res_vanilla = res_df[woe.stats.columns].copy()
print('Размерность_класс:', res_class.shape)
print('Размерность_vanilla:', res_vanilla.shape)

# Проверка результатов
print('Всего ячеек в результатах:', 
      res_class.shape[0]*res_class.shape[1],
      res_vanilla.shape[0]*res_vanilla.shape[1])
print('Кол-во одинаковых ячеек:', (res_class.fillna('пусто') == res_vanilla).sum().sum())

# %% [markdown]
# Результаты группировки одинаковые

# %% [markdown]
# ## Сравнение применения группировки с классическим трансформером

# %% ExecuteTime={"end_time": "2020-04-28T06:16:08.485492Z", "start_time": "2020-04-28T06:16:08.474431Z"} scrolled=true
# Количество одинаковых ячеек по столбцам
(res_df_trained.T == fit_transformed_same).sum()

# %% [markdown]
# Результаты применения группировки к (тем же) данным одинаковые везде, кроме признака "nums_w_letters".  
#
# Это произошло из-за того, что в столбце содержались как числа, так и строки, и при создании датасета тип столбца не был явно указан как "object", из-за чего тип автоматически был выбран "str" и числа преобразовались в строки (их в датасете 9000 из 10000 значений)  
#
# Если явно привести тип к числовому (где это возможно) и повторить расчеты для этого столбца

# %% ExecuteTime={"end_time": "2020-04-28T06:16:10.067142Z", "start_time": "2020-04-28T06:16:08.486466Z"} scrolled=true
# Обучение и применение группировок классическим вариантом
res_df_0 = pd.DataFrame()
res_df_trained_0 = pd.DataFrame()
for col in df_0[['nums_w_letters']]:
    print(col)
    cv = ['пусто']
    if 'nums_w_letters' in col:
        cv.extend(['d', 'f'])

    t = woe_vanilla.woeTransformer(df_0[col].fillna('пусто'),
                                   df_0['target'], plot=True, cat_values=cv)
    o = woe_vanilla.woe_apply(df_0[col].apply(pd.to_numeric, errors='ignore').fillna('пусто'), t)
    o.name = col
    t.insert(1, 'predictor', col)
    res_df_0 = res_df_0.append(t)
    res_df_trained_0 = res_df_trained_0.append(o)

# %% ExecuteTime={"end_time": "2020-04-28T06:16:10.074350Z", "start_time": "2020-04-28T06:16:10.068332Z"} scrolled=true
# Количество одинаковых ячеек по столбцам
(res_df_trained_0.T == fit_transformed_same[['nums_w_letters']]).sum()

# %% [markdown]
# ## Сравнение группировки с улучшенным трансформером

# %% ExecuteTime={"end_time": "2020-04-28T06:16:13.652352Z", "start_time": "2020-04-28T06:16:10.075352Z"}
# Обучение и применение группировок классическим вариантом
res_df = pd.DataFrame()
res_df_trained = pd.DataFrame()
for col in df_0.columns[:-1]:
    print(col)
    cv = ['пусто']
    if 'nums_w_letters' in col:
        cv.extend(['d', 'f'])
    elif 'letters' in col:
        cv.extend(df_0[col].unique())

    t = woe_new.woeTransformer(df_0[col].fillna('пусто'), df_0['target'], plot=True, cat_values=cv)
    o = woe_new.woe_apply(df_0[col].apply(pd.to_numeric, errors='ignore').fillna('пусто'), t)
    o.name = col
    t.insert(1, 'predictor', col)
    res_df = res_df.append(t)
    res_df_trained = res_df_trained.append(o)

# %% ExecuteTime={"end_time": "2020-04-28T06:16:13.666466Z", "start_time": "2020-04-28T06:16:13.652352Z"} scrolled=true
# результаты группировки классом
res_class = woe.stats.copy()
res_new = res_df[woe.stats.columns].copy()
print('Размерность_класс:', res_class.shape)
print('Размерность_vanilla:', res_vanilla.shape)

# Проверка результатов
print('Всего ячеек в результатах:', 
      res_class.shape[0]*res_class.shape[1],
      res_new.shape[0]*res_new.shape[1])
print('Кол-во одинаковых ячеек:', (res_class.fillna('пусто') == res_new).sum().sum())

# %% [markdown]
# Результаты группировки одинаковые

# %% [markdown]
# ## Сравнение применения группировки с улучшенным трансформером

# %% ExecuteTime={"end_time": "2020-04-28T06:16:13.682483Z", "start_time": "2020-04-28T06:16:13.668472Z"} scrolled=true
# Количество одинаковых ячеек по столбцам
(res_df_trained.T == fit_transformed_same).sum()
