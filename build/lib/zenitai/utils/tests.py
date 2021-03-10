"""
Модуль с функциями для тестирования
"""
import time

import numpy as np
import pandas as pd


def compare_results_test(fun1, fun2, exact=True, **kwargs):
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
    level = "serious"
    try:
        if type(res1) == pd.DataFrame and type(res2) == pd.DataFrame:
            assert res1.shape == res2.shape, "Разное кол-во строк/столбцов"
            assert np.round(res1.sum().sum(), 0) == np.round(
                res2.sum().sum(), 0
            ), "Суммы значений не одинаковы при округлении до целых"

            assert np.round(res1.sum().sum(), 4) == np.round(
                res2.sum().sum(), 4
            ), "Суммы значений не одинаковы при округлении до 4 знаков"

            level = "mild"

            assert np.round(res1.sum().sum(), 2) == np.round(
                res2.sum().sum(), 2
            ), "Суммы значений не одинаковы при округлении до 2 знаков"

            assert (
                ((res1.round(4) == res2.round(4)).sum().sum()) == np.multiply(*res1.shape) == np.multiply(*res2.shape)
            ), "Какие-то из ячеек результата различны (после округления до 4 знаков)"
            if exact:
                level = "minor"
                assert (
                    ((res1 == res2).sum().sum()) == np.multiply(*res1.shape) == np.multiply(*res2.shape)
                ), "Какие-то из ячеек результата различны (без округления)"
        else:
            assert len(res1) == len(res2)
            level = "mild"
            assert sum(res1) == sum(res2)
            if exact:
                level = "minor"
                assert all([i == j for i, j in zip(res1, res2)])
        print("Все тесты результатов пройдены: результаты функций одинаковы")
    except AssertionError as e:
        print(f"ОШИБКА ({level})!:", e)

    return [np.sum(np.sum(res1)), np.sum(np.sum(res2))]


def compare_time_test(fun1, fun2, n_iter=100, *args, **kwargs):
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

    time_recs = {fun1.__name__: [], fun2.__name__: []}
    time_aggs = {fun1.__name__: {"mean": np.nan, "std": np.nan}, fun2.__name__: {"mean": np.nan, "std": np.nan}}
    try:
        for f, fun_str in zip([fun1, fun2], list(time_recs.keys())):
            for i in range(n_iter):
                start_time = time.time()
                _ = f(**kwargs)
                time_recs[fun_str].append(time.time() - start_time)
            time_aggs[fun_str].update(
                {"mean": np.round(np.mean(time_recs[fun_str]), 10), "std": np.round(np.std(time_recs[fun_str]), 10)}
            )
        print("Замеры времени успешно завершены")
    except Exception:
        print("Замер времени не удался")

    return time_aggs
