"""
Модуль с полезными общими функциями (не специфичными для конкретной задачи или обработки)
Например: поиск пересекающихся значений в двух pandas.Series, применение стиля к pandas.DataFrame
"""
import numpy as np
import pandas as pd
from functools import partial

# %%


def generate_train_data(seed=42):
    np.random.seed(seed)
    digit_weighs = np.ones(10) / 10
    unequal_frequencies = [0.6, 0.3, 0.1]
    small_cat_size = 10
    major_part = 0.9
    minor_part = 1 - major_part

    res = pd.DataFrame({
        'digits': np.random.choice(range(10), 10000),
        'integers': np.random.choice(range(100), 10000),
        'floats': np.random.choice(np.linspace(1, 10, 100), 10000),
        'large_floats': np.random.choice(np.linspace(1, 10**6, 10*3), 10000),
        'integers_w_neg': np.random.choice(range(-50, 51), 10000),
        'integers_w_small_cat': np.hstack((np.array(np.random.choice(range(10), 9990), float),
                                           [100]*10)),
        'floats_w_na': np.hstack((np.array(np.random.choice(range(10), 9000), float),
                                  np.full(1000, np.nan))),
        'integers_w_letters': np.hstack((np.random.choice(range(10), 9000),
                                         np.array(['d', 'f']*500))),
        'integers_w_letters_obj': np.hstack((np.array(np.random.choice(range(10), 9000), int).astype(str),
                                             np.array(['d', 'f']*500, object))),
        'letters': np.random.choice(['a', 'b', 'c', 'd', 'e', 'f', 'g'], 10000),
        'letters_uneq_freq': np.random.choice(['a', 'b', 'c'], 10000, p=[0.6, 0.3, 0.1]),
        'letters_w_na': np.hstack((np.array(np.random.choice(['a', 'b', 'c'], 9000), object),
                                   np.full(1000, np.nan))),
        'single_integer': np.ones(10000),
        'single_letter': np.array(['x']*10000),
        'single_nan': np.full(10000, np.nan),
        'target': np.random.choice(range(2), 10000, p=[0.95, .05])})

    return res


def generate_test_data():
    digit_weighs = [.15, .15, .05, .05, .1, .1, .1, .01, .19, .1]
    unequal_frequencies = [0.1, 0.6, 0.3]
    small_cat_size = 10
    major_part = 0.8
    minor_part = 1 - major_part

    res = pd.DataFrame({
        'digits': np.random.choice(range(10), 10000,
                                   p=[.15, .15, .05, .05, .1, .1, .1, .01, .19, .1]),  # изменены веса
        'integers': np.random.choice(range(100), 10000),
        'floats': np.random.choice(np.linspace(1, 20, 100), 10000),
        'large_floats': np.random.choice(np.linspace(1, 10**6, 10*3), 10000),
        # изменен диапазон значений
        'integers_w_neg': np.random.choice(range(-100, 101), 10000),
        'integers_w_small_cat': np.hstack((np.array(np.random.choice(range(10), 9990), float),
                                           [100]*10)),
        'floats_w_na': np.hstack((np.array(np.random.choice(range(10), 8000), float),
                                  np.full(2000, np.nan))),                # увеличено кол-во пустышек
        'integers_w_letters': np.hstack((np.random.choice(range(10), 9000),
                                         np.array(['d', 'X']*500))),                 # одна категория заменена
        'integers_w_letters_obj': np.hstack((np.array(np.random.choice(range(10), 9000), int).astype(str),
                                             np.array(['d', 'f', 'y', 'z']*250, object))),   # добавлены 2 новых
        'letters': np.random.choice(['a', 'b', 'c', 'd', 'e', 'f', 'g'], 10000),
        # изменены веса
        'letters_uneq_freq': np.random.choice(['a', 'b', 'c'], 10000, p=[0.1, 0.6, 0.3]),
        'letters_w_na': np.hstack((np.array(np.random.choice(['a', 'b', 'c', 'd'], 9000), object),
                                   np.full(1000, np.nan))),  # добавлена категория
        'letters_crazy': ['XYZ']*10000,
        'single_integer': np.ones(10000),
        'single_letter': np.array(['x']*10000),
        'single_nan': np.full(10000, np.nan),
        'target': np.random.choice(range(2), 10000, p=[0.95, .05])})

    return res
