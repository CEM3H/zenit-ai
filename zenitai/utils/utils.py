"""
Модуль с полезными общими функциями (не специфичными для конкретной задачи или обработки)
Например: поиск пересекающихся значений в двух pandas.Series, применение стиля к pandas.DataFrame
"""
import numpy as np
import pandas as pd


def read_from_mssql(path, **kwargs):
    df = pd.read_csv(path, encoding="utf-8", sep=";", dtype="object", **kwargs)
    print(df.shape)
    return df


# Comes from 1. Superscore_Zenit_Features.ipynb, cell
def compare_series(s1, s2, ret_index=False):
    """Сравнивает два объекта pd.Series на предмет наличия совпадающих
    элементов
    Вход:
        - s1, s2 : pd.Series, объекты для сравнения
        - ret_index : bool, модификатор вывода результатов
                        * True - возвращаются только
    Выход:
        - общие элементы из s1
        - общие элементы из s2
        - элементы, уникальные для s1
        - элементы, уникальные для s2
    """
    assert type(s1) == pd.Series
    assert type(s2) == pd.Series

    s1_common_elems = s1[s1.isin(s2)]
    s2_common_elems = s2[s2.isin(s1)]
    s1_only = s1[~s1.isin(s2)]
    s2_only = s2[~s2.isin(s1)]

    return s1_common_elems, s2_common_elems, s1_only, s2_only


#%%
def generate_train_data(seed=42):
    np.random.seed(seed)
    digit_weighs = np.ones(10) / 10
    unequal_frequencies = [0.6, 0.3, 0.1]
    small_cat_size = 10
    major_part = 0.9
    minor_part = 1 - major_part

    a = np.random.rand(2500) * 100

    res = pd.DataFrame(
        {
            "digits": np.random.choice(range(10), 10000),
            "integers": np.random.choice(range(100), 10000),
            "many_unique_floats": np.hstack((a, a, a, a)),
            "all_unique_floats": np.random.rand(10000) * 100,
            "many_floats_w_cat": np.hstack((np.random.rand(5000) * 100, np.array(["d", "f"] * 2500))),
            "many_floats_w_cat_and_nan": np.hstack((np.random.rand(7000) * 100, np.array(["d", "f", np.nan] * 1000))),
            "many_floats_w_outliers": np.hstack((np.random.rand(5000) * 100, np.array([-10000, 10000] * 2500))),
            "floats": np.random.choice(np.linspace(1, 10, 100), 10000),
            "large_floats": np.random.choice(np.linspace(1, 10 ** 6, 10 * 3), 10000),
            "integers_w_neg": np.random.choice(range(-50, 51), 10000),
            "integers_w_small_cat": np.hstack((np.array(np.random.choice(range(10), 9990), float), [100] * 10)),
            "floats_w_na": np.hstack((np.array(np.random.choice(range(10), 9000), float), np.full(1000, np.nan))),
            "integers_w_letters": np.hstack((np.random.choice(range(10), 9000), np.array(["d", "f"] * 500))),
            "integers_w_letters_obj": np.hstack(
                (np.array(np.random.choice(range(10), 9000), int).astype(str), np.array(["d", "f"] * 500, object))
            ),
            "letters": np.random.choice(["a", "b", "c", "d", "e", "f", "g"], 10000),
            "letters_uneq_freq": np.random.choice(["a", "b", "c"], 10000, p=[0.6, 0.3, 0.1]),
            "letters_w_na": np.hstack(
                (np.array(np.random.choice(["a", "b", "c"], 9000), object), np.full(1000, np.nan))
            ),
            "single_integer": np.ones(10000),
            "single_letter": np.array(["x"] * 10000),
            "single_nan": np.full(10000, np.nan),
            "target": np.random.choice(range(2), 10000, p=[0.95, 0.05]),
        }
    )

    return res


def generate_test_data():
    digit_weighs = [0.15, 0.15, 0.05, 0.05, 0.1, 0.1, 0.1, 0.01, 0.19, 0.1]
    unequal_frequencies = [0.1, 0.6, 0.3]
    small_cat_size = 10
    major_part = 0.8
    minor_part = 1 - major_part

    res = pd.DataFrame(
        {
            "digits": np.random.choice(
                range(10), 10000, p=[0.15, 0.15, 0.05, 0.05, 0.1, 0.1, 0.1, 0.01, 0.19, 0.1]
            ),  # изменены веса
            "integers": np.random.choice(range(100), 10000),
            "floats": np.random.choice(np.linspace(1, 20, 100), 10000),
            "large_floats": np.random.choice(np.linspace(1, 10 ** 6, 10 * 3), 10000),
            "integers_w_neg": np.random.choice(range(-100, 101), 10000),  # изменен диапазон значений
            "integers_w_small_cat": np.hstack((np.array(np.random.choice(range(10), 9990), float), [100] * 10)),
            "floats_w_na": np.hstack(
                (np.array(np.random.choice(range(10), 8000), float), np.full(2000, np.nan))
            ),  # увеличено кол-во пустышек
            "integers_w_letters": np.hstack(
                (np.random.choice(range(10), 9000), np.array(["d", "X"] * 500))
            ),  # одна категория заменена
            "integers_w_letters_obj": np.hstack(
                (
                    np.array(np.random.choice(range(10), 9000), int).astype(str),
                    np.array(["d", "f", "y", "z"] * 250, object),
                )
            ),  # добавлены 2 новых
            "letters": np.random.choice(["a", "b", "c", "d", "e", "f", "g"], 10000),
            "letters_uneq_freq": np.random.choice(["a", "b", "c"], 10000, p=[0.1, 0.6, 0.3]),  # изменены веса
            "letters_w_na": np.hstack(
                (np.array(np.random.choice(["a", "b", "c", "d"], 9000), object), np.full(1000, np.nan))
            ),  # добавлена категория
            "letters_crazy": ["XYZ"] * 10000,
            "single_integer": np.ones(10000),
            "single_letter": np.array(["x"] * 10000),
            "single_nan": np.full(10000, np.nan),
            "target": np.random.choice(range(2), 10000, p=[0.95, 0.05]),
        }
    )

    return res
