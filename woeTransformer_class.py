"""
WOE-трансформер в виде класса с интерфейсом как в sklearn - с методами fit 
и predict
"""
import math
import time

import numpy as np
import pandas as pd
import seaborn as sns

class WoeTransformer:
    class GroupedPredictor(pd.DataFrame):
        def get_predictor(self, x):
            return self[self['predictor']==x]
    
    
    def __init__(self, min_sample_rate=0.05, min_count=3, detect_categories=False):
        self.min_sample_rate = min_sample_rate
        self.min_count = min_count
        self.detect_categories = detect_categories
    
    def _grouping(self, X, y, low_accuracy=None):
        """
        Агрегация данных по значениям предиктора. Рассчитывает количество наблюдений,
        количество целевых событий, долю группы от общего числа наблюдений и долю целевых в группе 

        Входные данные:
        ---------------
            DF_data_i : pandas.DataFrame
                    Таблица данных для агрегации, должна содержать поля 'predictor' и 'target'. 
                    Поле target при этом должно состоять из 0 и 1, где 1 - целевое событие
            low_acc : int, default None
                    Параметр для округления значений предиктора. 
                    Если None, то предиктор не округляется.
                    Если целое неотрицательное число, параметр используется для определения 
                    количества знаков после запятой, остальные значения игнорируются

        Возвращает:
        ---------------
            DF_grouping : pandas.DataFrame
                    Таблица с агрегированными данными по значениям предиктора
                
        """
        # Округление, если аргумент принимает допустимые значения
        if low_accuracy and type(low_accuracy) is int and low_accuracy > 0:
            df = X.round(low_accuracy).copy()
        else:
            df = X.copy()
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
        self.grouped = GroupedPredictor(self.grouped)
    
    def _calc_trend_coefs(self, x, y):
            return {x.name:np.polyfit(x, y, deg=1)}

    def fit(self, X, y, low_accuracy=None):#, cat_values:dict={}):
        self._grouping(X, y, low_accuracy)
        return self._fit_numeric(X, y)
    
    
    def _fit_numeric(self, X, y):
        self.trend_coefs = {}
        self.borders = {}
        for i in X:
            self.trend_coefs.update(self._calc_trend_coefs(X[i], y))
            self.borders.update({i:self._monotonic_borders(self.grouped.get_predictor(i), 
                                                           self.trend_coefs[i])})

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
            min_sample_rate : float, default 0.05 
                    Минимальный размер группы (доля от размера выборки)
            min_count : int, default 3
                    Минимальное количество наблюдений каждого класса в группе

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
        return R_borders





woe = WoeTransformer()
print(woe.min_sample_rate)
print(woe.min_count)
print(woe.detect_categories)
