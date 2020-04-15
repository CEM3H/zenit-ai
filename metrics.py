"""
Модуль с функциями для расчета метрик - Gini, PSI и т.п.
"""
import pandas as pd
import numpy as np


def calcPSI(exp, act):
    """
    Расчет значений PSI для сравнения распределений одной переменной в двух выборках
    Предполагается, что количественные переменные будут представлены в виде
    соответствующих групп (после равномерного биннинга или WOE-преобразования).
    Категориальные переменные могут передаваться без предварительных трансформаций

    Входные данные:
            exp : pandas.Series
                    Значения предиктора из первой выборки ("ожидаемые" в терминологии PSI)
            act : pandas.Series
                    Значения предиктора из второй выборки ("наблюдаемые" в терминологии PSI)
    Возвращает:
            df : pandas.DataFrame
                    Таблица с ожидаемыми и наблюдаемыми частотами и рассчитанных PSI по каждой группе

    """
    # Расчет долей каждой категории в обеих выборках
    exp = exp.value_counts(normalize=True).sort_index()
    act = act.value_counts(normalize=True).sort_index()
    # Соединение в один датафрейм
    df = pd.concat([exp, act], axis=1).fillna(0).reset_index()
    df.columns = ['group', 'expected', 'actual']
    # Расчет PSI по каждой группе
    df['PSI'] = ((df['actual'] - df['expected'])
                 * np.log((df['actual'] + 0.000001) / (df['expected'] + 0.000001)))

    return df
