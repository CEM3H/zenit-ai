"""
Модуль с функциями для тестирования
"""
from utils import generate_train_data
import time
import random
import numpy as np
import pandas as pd


def compareResultsTest(fun1, fun2, exact=True, **kwargs):
    """
    Сравнение результатов двух функций, которые возвращают pd.DataFrame или list

    Входные данные:
    ---------------
            fun1, fun2 : function
                    Функции, результат которых следует сравнить
            exact : bool, default True
                    Флаг для проверки точных значений.
                    Если выключен: проверка точности только до 2 знака после запятой
            **kwargs
                    Аргументы, которые принимают функции
                    Тестируемые функции должны принимать одинаковые аргументы
    Возвращает:
    -----------
            list of len 2
                    Элементы списка - сумма всех значений результата каждой функции

    """
    res1 = fun1(**kwargs)
    res2 = fun2(**kwargs)
    level = 'serious'
    try:
        if type(res1) == pd.DataFrame and type(res2) == pd.DataFrame:
            assert res1.shape == res2.shape, 'Разное кол-во строк/столбцов'
            assert np.round(res1.sum().sum(), 0) == np.round(res2.sum().sum(),
                                                             0), 'Суммы значений не одинаковы при округлении до целых'

            assert np.round(res1.sum().sum(), 4) == np.round(res2.sum().sum(), 4), \
                'Суммы значений не одинаковы при округлении до 4 знаков'

            level = 'mild'

            assert np.round(res1.sum().sum(), 2) == np.round(res2.sum().sum(), 2), \
                'Суммы значений не одинаковы при округлении до 2 знаков'

            assert (((res1.round(4) == res2.round(4)).sum().sum())
                    == np.multiply(*res1.shape)
                    == np.multiply(*res2.shape)), 'Какие-то из ячеек результата различны (после округления до 4 знаков)'
            if exact:
                level = 'minor'
                assert (((res1 == res2).sum().sum())
                        == np.multiply(*res1.shape)
                        == np.multiply(*res2.shape)), 'Какие-то из ячеек результата различны (без округления)'
        else:
            assert len(res1) == len(res2)
            level = 'mild'
            assert sum(res1) == sum(res2)
            if exact:
                level = 'minor'
                assert all([i == j for i, j in zip(res1, res2)])
        print('Все тесты результатов пройдены: результаты функций одинаковы')
    except AssertionError as e:
        print(f'ОШИБКА ({level})!:', e)

    return [np.sum(np.sum(res1)), np.sum(np.sum(res2))]


def compareTimeTest(fun1, fun2, n_iter=100, *args, **kwargs):
    """
    Сравнение времени работы двух функций

    Входные данные:
    ---------------
            fun1, fun2 : function
                    Функции, время работы которых следует сравнить
            n_iter : int, default 100
                    Количество прогонов каждой функции
            **kwargs
                    Аргументы, которые принимают функции
                    Тестируемые функции должны принимать одинаковые аргументы
    Возвращает:
    -----------
            time_aggs : dict
                    Словарь словарей с информацией о среднем значении и стандартным отклонением времени выполнения
                    Ключи верхнего уровня - имена тестируемых функций

    """

    time_recs = {fun1.__name__: [],
                 fun2.__name__: []}
    time_aggs = {fun1.__name__: {'mean': np.nan,
                                 'std': np.nan
                                 },
                 fun2.__name__: {'mean': np.nan,
                                 'std': np.nan
                                 }
                 }
    try:
        for f, fun_str in zip([fun1, fun2], list(time_recs.keys())):
            for i in range(n_iter):
                start_time = time.time()
                _ = f(**kwargs)
                time_recs[fun_str].append(time.time() - start_time)
            time_aggs[fun_str].update({'mean': np.round(np.mean(time_recs[fun_str]), 10),
                                       'std':  np.round(np.std(time_recs[fun_str]),  10)})
        print('Замеры времени успешно завершены')
    except:
        print('Замер времени не удался')

    return time_aggs


def dataframe_equal_test(df1, df2):
    df1 = df1.sort_index()
    df1.columns = df1.columns.str.lower()
    df1 = df1[sorted(df1.columns)].fillna(-999)

    df2 = df2.sort_index()
    df2.columns = df2.columns.str.lower()
    df2 = df2[sorted(df2.columns)]#.fillna(-999)
    df2.fillna(-999, inplace=True)
    # Columns and index test
    shape_test = df1.shape == df2.shape
    col_test = all(df1.columns == df2.columns)
    ind_test = all(df1.index == df2.index)

    if col_test and ind_test and shape_test:
        for i in range(10, -1, -1):
            eq_vals = (df1.round(i) == df2.round(i)).sum().sum()
            if eq_vals == np.multiply(*df1.shape):
                return "Dataframe's values all equal at {} digits precision".format(i)
            elif i == 0:
                return "Dataframe's values are not equal"
    else:
        return "Dataframes are not labelled equally"


t1 = generate_train_data(42)
t2 = generate_train_data()

print(dataframe_equal_test(t1, t2.iloc[:-1]))
