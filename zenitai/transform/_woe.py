"""
Функции и классы для проведения WoE-преобразований
"""


import math
import warnings

import numpy as np
import pandas as pd
import sklearn as sk
from IPython.display import display
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


class _GroupedPredictor(pd.DataFrame):
    """
    Вспомогательный класс для удобства доступа к некоторым данным

    """

    def get_predictor(self, x):
        """
        Получение подвыборки по имени предиктора(ов)

        Parameters
        ---------------
            x : str/int/list-like
                    Предиктор или список предикторов

        Returns:
        -----------
            self : pd.DataFrame
                    Часть датафрейма (самого себя)

        """

        if isinstance(x, (list, set, tuple)):
            return self[self["predictor"].isin(x)]
        else:
            return self[self["predictor"] == x]

    def append(self, other):
        return _GroupedPredictor(super().append(other))


class WoeTransformer(TransformerMixin, BaseEstimator):
    """Класс для построения и применения WOE группировки к датасету

    Parameters
    ----------
    min_sample_rate : float, default 0.05
        Минимальный размер группы (доля от размера выборки)
    min_count : int, default 3
        Минимальное количество наблюдений каждого класса в группе
    save_data : bool, default False
        Параметр, определяющий, нужно ли сохранить данные для обучения
        трансформера внутри экземпляра класса
    join_bad_categories : bool, default False
        Определяет, должени ли трансформер предпринять попытку для объединения
        катогориальных групп в более крупные

        Warning
        -------
        join_bad_categories - Экспериментальная функция.
        Способ группировки категорий нестабилен

    Attributes
    ----------
    stats : pandas.DataFrame
        Результаты WOE-группировки по всем предикторам

    predictors : list
        Список предикторов, на которых была построена группировка

    cat_values : dict[str, list]
        Словарь со списками категорий по предикторам, переданный при обучении

    alpha_values : dict[str, float]
        Словарь со значениями alpha для регуляризации групп

    possible groups : pandas.DataFrame
        Данные о значениях предиктора, которые могли бы стать
        отдельными категориями

    bad_groups : pandas.DataFrame
        Данные о группах, которые не удовлетворяют условиям

    """

    def __repr__(self):
        return "WoeTransformer(min_sample_rate={!r}, min_count={!r}, n_fitted_predictors={!r})".format(
            self.min_sample_rate,
            self.min_count,
            len(self.predictors),
        )

    def __init__(
        self,
        min_sample_rate: float = 0.05,
        min_count: int = 3,
        save_data: bool = False,
        join_bad_categories: bool = False,
    ):
        """
        Инициализация экземпляра класса

        """
        self.min_sample_rate = min_sample_rate
        self.min_count = min_count
        self.predictors = []
        self.alpha_values = {}
        self.save_data = save_data
        self.join_bad_categories = join_bad_categories

    # -------------------------
    # Функции интерфейса класса
    # -------------------------

    def fit(self, X, y, cat_values={}, alpha_values={}):
        """
        Обучение трансформера и расчет всех промежуточных данных

        Parameters
        ---------------
        X : pd.DataFrame
                Датафрейм с предикторами, которые нужно сгруппировать
        y : pd.Series
                Целевая переменная
        cat_values : dict[str, list[str]], optional
                Словарь списков с особыми значениями, которые нужно
                выделить в категории
                По умолчанию все строковые и пропущенные значения
                выделяются в отдельные категории
        alpha_values : dict[str, float], optional
                Словарь со значениями alpha для регуляризации WOE-групп

        Returns
        -------
        self : WoeTransformer

        """
        # Сброс текущего состояния трансформера
        self._reset_state()
        # Сохранение категориальных значений
        self.cat_values = cat_values
        # Валидация данных и решейпинг
        if hasattr(self, "_validate_data"):
            X, y = self._validate_and_convert_data(X, y)
        if self.save_data:
            self.data = X
            self.target = y
        # Инициализация коэффициентов для регуляризации групп
        self.alpha_values = {i: 0 for i in X.columns}
        self.alpha_values.update(alpha_values)

        # Агрегация значений предикторов
        self._grouping(X, y)
        # Расчет WOE и IV
        self._fit_numeric(X, y)
        # Поиск потенциальных групп
        # Поиск "плохих" групп
        self._get_bad_groups()

        return self

    def transform(self, X, y=None):
        """
        Применение обученного трансформера к новым данным

        Parameters
        ---------------
        X : pandas.DataFrame
                Датафрейм, который нужно преобразовать
                Предикторы, которые не были сгруппированы ранее, будут
                проигнорированы и выведется сообщение
        y : pandas.Series
                Игнорируется

        Returns
        -----------
        transformed : pandas.DataFrame
                Преобразованный датасет

        """
        transformed = pd.DataFrame()
        if hasattr(self, "_validate_data"):
            try:
                X, y = self._validate_and_convert_data(X, y)
            except AttributeError:
                pass
        for i in X:
            if i in self.predictors:
                try:
                    transformed[i] = self._transform_single(X[i])
                except Exception as e:
                    print(f"Transform failed on predictor: {i}", e)
            else:
                print(f"Column is not in fitted predictors list: {i}")
        return transformed

    def fit_transform(self, X, y, cat_values={}, alpha_values={}):
        """
        Обучение трансформера и расчет всех промежуточных данных
        с последующим примененим группировки к тем же данным

        Parameters
        ---------------
        X : pandas.DataFrame
                Датафрейм с предикторами, которые нужно сгруппировать
        y : pandas.Series
                Целевая переменная
        cat_values : dict[str, list[str]], optional
                Словарь списков с особыми значениями, которые нужно
                выделить в категории
                По умолчанию все строковые и пропущенные значения
                выделяются в отдельные категории
        alpha_values : dict[str, float], optional
                Словарь со значениями alpha для регуляризации WOE-групп

        Returns
        -----------
        transformed : pd.DataFrame
                Преобразованный датасет

        """

        self.fit(X, y, cat_values=cat_values, alpha_values=alpha_values)
        return self.transform(X)

    def plot_woe(self, predictors=None):
        """
        Отрисовка одного или нескольких графиков группировки

        Parameters
        ---------------
        predictors : str or array, default None
                Предиктор(ы), по которым нужны графики
                -- если str - отрисовывается один график
                -- если array - отрисовываются графики из списка
                -- если None - отрисовываются все сгруппированные предикторы
        Warning
        -------
        Запуск метода без аргументов может занять длительное время при большом
        количестве предикторов
        """

        if predictors is None:
            predictors = self.predictors
        elif isinstance(predictors, str):
            predictors = [predictors]
        elif isinstance(predictors, (list, tuple, set)):
            predictors = predictors

        _, axes = plt.subplots(figsize=(10, len(predictors) * 5), nrows=len(predictors))
        try:
            for i, col in enumerate(predictors):
                self._plot_single_woe_grouping(self.stats.get_predictor(col), axes[i])
        except TypeError:
            self._plot_single_woe_grouping(self.stats.get_predictor(col), axes)

        # return fig

    def get_iv(self, sort=False):
        """Получение списка значений IV по предикторам
        Parameters
        ----------
        sort : bool, default False
            Включает сортировку результата по убыванию IV

        Returns
        -------
        pandas.Series

        """
        try:
            res = self.stats.groupby("predictor")["IV"].sum()
            if sort:
                res = res.sort_values(ascending=False)
            res = dict(res)
        except AttributeError as e:
            print(f"Transformer was not fitted yet. {e}")
            res = {}

        return res

    # -------------------------
    # Внутренние функции над всем датасетом
    # -------------------------

    def _validate_and_convert_data(self, X, y):
        """Проверяеn входные данные, трансформирует в объекты pandas
        Использует метод _validate_data из sklearn/base.py
        """

        if hasattr(X, "columns"):
            predictors = X.columns
        else:
            predictors = ["X" + str(i + 1) for i in range(X.shape[1])]
        if y is None:
            X_valid = self._validate_data(X, y, dtype=None, force_all_finite=False)
            X_valid = pd.DataFrame(X, columns=predictors)
            y_valid = None
        else:
            X_valid, y_valid = self._validate_data(
                X, y, dtype=None, force_all_finite=False
            )
            y_valid = pd.Series(y, name="target")
            X_valid = pd.DataFrame(X, columns=predictors)

        return X_valid, y_valid

    def _grouping(self, X, y):
        """
        Применение группировки ко всем предикторам
        """
        df = X.copy()
        df = df.fillna("пусто")
        df["target"] = y.copy()

        # Группировка и расчет показателей
        for col in df.columns[:-1]:
            grouped_temp = self._group_single(df[col], y)
            num_mask = self._get_nums_mask(grouped_temp["value"])
            cat_val_mask = grouped_temp["value"].isin(self.cat_values.get(col, []))
            is_all_categorical = all(~num_mask | cat_val_mask)

            if self.join_bad and is_all_categorical:
                repl = self._get_cat_values_for_join(grouped_temp)
                grouped_temp = self._group_single(df[col].replace(repl), y)
            self.grouped = self.grouped.append(grouped_temp)

        # Замена пустых значений обратно на np.nan ИЛИ преобразование в числовой тип
        try:
            self.grouped["value"] = self.grouped["value"].replace({"пусто": np.nan})
        except TypeError:
            self.grouped["value"] = pd.to_numeric(
                self.grouped["value"], downcast="signed"
            )

    def _fit_numeric(self, X, y):
        """
        Расчет WOE и IV

        Parameters:
        ---------------
        X : pd.DataFrame
                Датафрейм с предикторами, которые нужно сгруппировать
        y : pd.Series
                Целевая переменная
        Returns
        -------
        None

        """

        res = pd.DataFrame()

        for i in X:
            res_i = self._fit_single(X[i], y)
            res = res.append(res_i)
            self.predictors.append(i)
        self.stats = self.stats.append(res)

    # -------------------------
    # Внутренние функции над отдельными столбцами
    # -------------------------
    def _group_single(self, x, y):
        """
        Агрегация данных по значениям предиктора.
        Рассчитывает количество наблюдений,
        количество целевых событий, долю группы от общего числа наблюдений
        и долю целевых в группе

        Parameters:
        ---------------
        X : pandas.DataFrame
                Таблица данных для агрегации
        y : pandas.Series
                Целевая переменная

        """
        col = x.name
        df = pd.DataFrame({col: x.values, "target": y.values})

        grouped_temp = df.groupby(col)["target"].agg(["count", "sum"]).reset_index()
        grouped_temp.columns = ["value", "sample_count", "target_count"]
        grouped_temp["sample_rate"] = (
            grouped_temp["sample_count"] / grouped_temp["sample_count"].sum()
        )
        grouped_temp["target_rate"] = (
            grouped_temp["target_count"] / grouped_temp["sample_count"]
        )
        grouped_temp.insert(0, "predictor", col)

        return _GroupedPredictor(grouped_temp)

    def _fit_single(self, x, y, gr_subset=None, cat_vals=None):
        """
        Расчет WOE и IV

        Parameters:
        ---------------
            X : pd.DataFrame
                    Датафрейм с предикторами, которые нужно сгруппировать
            y : pd.Series
                    Целевая переменная
            gr_subset : _GroupedPredictor
                    Предиктор
        """
        gr_subset_num = pd.DataFrame()
        gr_subset_cat = pd.DataFrame()
        col = x.name
        if gr_subset is None:
            gr_subset = self.grouped.get_predictor(col)
        if cat_vals is None:
            cat_vals = self.cat_values.get(col, [])
        nan_mask = x.isna()
        num_mask = self._get_nums_mask(x) & (~x.isin(cat_vals)) & (~nan_mask)
        num_vals = x.loc[num_mask].unique()

        try:
            # Расчет коэффициентов тренда по числовым значениям предиктора
            if num_mask.sum() > 0:
                try:
                    poly_coefs = np.polyfit(
                        x.loc[num_mask].astype(float), y.loc[num_mask], deg=1
                    )
                except np.linalg.LinAlgError as e:
                    print(f"Error in np.polyfit on predictor: '{col}'.\nError MSG: {e}")
                    print("Linear Least Squares coefficients were set to [1, 0]")
                    poly_coefs = np.array([1, 0])

                self.trend_coefs.update({col: poly_coefs})
                # Расчет монотонных границ
                gr_subset_num = gr_subset[gr_subset["value"].isin(num_vals)].copy()
                gr_subset_num["value"] = pd.to_numeric(gr_subset_num["value"])
                gr_subset_num = gr_subset_num.sort_values("value")
                borders = self._monotonic_borders(gr_subset_num, self.trend_coefs[col])
                self.borders.update({col: borders})
                # Применение границ к сгруппированным данным
                gr_subset_num["groups"] = pd.cut(gr_subset_num["value"], borders)
                gr_subset_num["type"] = "num"

        except ValueError as e:
            print(f"ValueError on predictor {col}.\nError MSG: {e}")

        # Расчет коэффициентов тренда по категориальным значениям предиктора
        if (~num_mask).sum() > 0:
            gr_subset_cat = gr_subset[~gr_subset["value"].isin(num_vals)].copy()
            gr_subset_cat["groups"] = gr_subset_cat["value"].fillna("пусто")
            gr_subset_cat["type"] = "cat"

        # Объединение числовых и категориальных значений
        gr_subset = pd.concat([gr_subset_num, gr_subset_cat], axis=0, ignore_index=True)

        # Расчет WOE и IV
        alpha = self.alpha_values.get(col, 0)
        res_i = self._statistic(gr_subset, alpha=alpha)
        is_empty_exists = any(res_i["groups"].astype(str).str.contains("пусто"))
        if is_empty_exists:
            res_i["groups"].replace({"пусто": np.nan}, inplace=True)

        return res_i

    def _transform_single(self, x, stats=None):
        """
        Применение группировки и WoE-преобразования

        Parameters
        ---------------
        x : pandas.Series
                Значения предиктора
        Returns
        ---------------
        X_woe : pandas.DataFrame
                WoE-преобразования значений предиктора
                WoE = 0, если группа не встречалась в обучающей выборке

        """
        orig_index = x.index
        X_woe = x.copy()
        if stats is None:
            stats = self.stats.get_predictor(X_woe.name)
        # Маппинги для замены групп на соответствующие значения WOE
        num_map = {
            stats.loc[i, "groups"]: stats.loc[i, "WOE"]
            for i in stats.index
            if stats.loc[i, "type"] == "num"
        }
        cat_map = {
            stats.loc[i, "groups"]: stats.loc[i, "WOE"]
            for i in stats.index
            if stats.loc[i, "type"] == "cat"
        }
        # Категориальные группы
        cat_bounds = stats.loc[stats["type"] == "cat", "groups"]

        # predict по числовым значениям
        DF_num = stats.loc[stats["type"] == "num"]
        if DF_num.shape[0] > 0:
            # Границы (правые) интервалов для разбивки числовых переменных
            num_bounds = [-np.inf] + list(
                pd.IntervalIndex(stats.loc[stats["type"] == "num", "groups"]).right
            )
            # Выделение только числовых значений предиктора
            # (похожих на числа и тех, что явно не указаны как категориальные)
            X_woe_num = pd.to_numeric(
                X_woe[(self._get_nums_mask(X_woe)) & (~X_woe.isin(cat_bounds))]
            )
            # Разбивка значений на интервалы в соответствии с группировкой
            X_woe_num = pd.cut(X_woe_num, num_bounds)
            # Замена групп на значения WOE
            X_woe_num = X_woe_num.replace(num_map)
            X_woe_num.name = "woe"
        else:
            X_woe_num = pd.Series()

        # predict по категориальным значениям (может обновлять значения по числовым)
        DF_cat = stats.loc[stats["type"] == "cat"]
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
        if len(used_index) < len(x):
            X_woe_oth = X_woe.index.drop(used_index)
            X_woe_oth = pd.Series(0, index=X_woe_oth)
        else:
            X_woe_oth = pd.Series()

        X_woe = pd.concat([X_woe_num, X_woe_cat, X_woe_oth]).reindex(orig_index)
        X_woe = pd.to_numeric(X_woe, downcast="signed")

        return X_woe

    def _monotonic_borders(self, grouped, p):
        """
        Определение оптимальных границ групп предиктора (монотонный тренд)

        Parameters
        ---------------
        DF_grouping : pandas.DataFrame
                Агрегированные данные по значениям предиктора (результат работы
                фунции grouping, очищенный от категориальных значений).
                Должен содержать поля 'predictor', 'sample_count', 'target_count',
                'sample_rate и 'target_rate'
        p : list-like, длиной в 2 элемента
                Коэффициенты линейного тренда значений предиктора

        Returns
        ---------------
        R_borders : list
                Правые границы групп для последующей группировки

        """
        k01, k11 = (1, 1) if p[0] > 0 else (0, -1)
        R_borders = []
        min_ind = 0  # минимальный индекс. Начальные условия

        DF_grouping = grouped.copy().sort_values("value").reset_index()

        while min_ind < DF_grouping.shape[0]:  # цикл по новым группам
            # Расчет показателей накопительным итогом
            DF_j = DF_grouping.iloc[min_ind:]
            DF_iter = DF_j[["sample_rate", "sample_count", "target_count"]].cumsum()
            DF_iter["non_target_count"] = (
                DF_iter["sample_count"] - DF_iter["target_count"]
            )
            DF_iter["target_rate"] = DF_iter["target_count"] / DF_iter["sample_count"]

            # Проверка на соответствие критериям групп
            DF_iter["check"] = self._check_groups(DF_iter)

            # Расчет базы для проверки оптимальности границы
            # В зависимости от тренда считается скользящий _вперед_ минимум или максимум
            # (в расчете участвуют все наблюдения от текущего до последнего)
            if k11 == 1:
                DF_iter["pd_gr"] = (
                    DF_iter["target_rate"][::-1]
                    .rolling(len(DF_iter), min_periods=0)
                    .min()[::-1]
                )
            else:
                DF_iter["pd_gr"] = (
                    DF_iter["target_rate"][::-1]
                    .rolling(len(DF_iter), min_periods=0)
                    .max()[::-1]
                )

            # Проверка оптимальности границы
            DF_iter["opt"] = DF_iter["target_rate"] == DF_iter["pd_gr"]
            DF_iter = pd.concat([DF_j[["value"]], DF_iter], axis=1)
            try:
                min_ind = DF_iter.loc[
                    (DF_iter["check"]) & (DF_iter["opt"]), "target_rate"
                ].index.values[0]
                score_j = DF_iter.loc[min_ind, "value"]
                if (
                    len(R_borders) > 0 and score_j == R_borders[-1]
                ):  # Выход из цикла, если нет оптимальных границ
                    break
            except Exception:
                break
            min_ind += 1
            R_borders.append(score_j)

        # Проверка последней добавленной группы
        if len(R_borders) > 0:
            DF_iter = DF_grouping.loc[DF_grouping["value"] > R_borders[-1]]
            sample_rate_i = DF_iter["sample_rate"].sum()  # доля выборки
            sample_count_i = DF_iter["sample_count"].sum()  # количество наблюдений
            target_count_i = DF_iter["target_count"].sum()  # количество целевых
            non_target_count_i = sample_count_i - target_count_i  # количество нецелевых

            if (
                (sample_rate_i < self.min_sample_rate)
                or (target_count_i < self.min_count)
                or (non_target_count_i < self.min_count)
            ):
                R_borders.remove(R_borders[-1])  # удаление последней границы
        else:
            predictor = DF_grouping["predictor"].iloc[0]
            warnings.warn(
                f"Couldn't find any borders for feature {predictor}.\n Borders set on (-inf, +inf)"
            )
        R_borders = [-np.inf] + R_borders + [np.inf]
        return R_borders

    def _check_groups(
        self,
        df,
        sample_rate_col="sample_rate",
        sample_count_col="sample_count",
        target_count_col="target_count",
    ):
        """ Проверить сгруппированные значения предиктора на соответствме условиям"""
        cond_mask = (
            (df[sample_rate_col] >= self.min_sample_rate - 10 ** -9)
            & (df[sample_count_col] >= self.min_count)
            & (df[target_count_col] >= self.min_count)
        )

        return cond_mask

    def _get_cat_values_for_join(self, grouped):
        """Получить словарь для замены категорий на объединяемые
        NOTE: Нужно тестирование
        TODO: переписать на рекурсию
        """
        df = grouped.copy()

        cond_mask = ~self._check_groups(df)

        res = df[
            [
                "predictor",
                "value",
                "sample_count",
                "target_count",
                "sample_rate",
                "target_rate",
            ]
        ].copy()
        res = res.sort_values(["sample_rate", "target_rate"])
        res["cum_sample_rate"] = res["sample_rate"].cumsum()
        res["check"] = cond_mask
        res["check_reverse"] = ~cond_mask
        res["check_diff"] = res["check"].astype(int).diff()
        res["new_group"] = (res["check_diff"] == -1).astype(int)
        res["exist_group"] = res["check_reverse"].astype(int).eq(1)
        res.loc[~res["check_reverse"], "exist_group"] = np.NaN
        res["exist_group_cum"] = res["exist_group"].cumsum().fillna(method="bfill")

        res[["cum_sr", "cum_sc", "cum_tc"]] = res.groupby("exist_group_cum").agg(
            {
                "sample_rate": "cumsum",
                "sample_count": "cumsum",
                "target_count": "cumsum",
            }
        )

        res["cum_sr_check"] = (
            self._check_groups(res, "cum_sr", "cum_sc", "cum_tc")
            .astype(int)
            .diff()
            .eq(1)
            .astype(int)
            .shift()
        )

        display(res)
        res.loc[res["cum_sr_check"] != 1, "cum_sr_check"] = np.nan
        res["cum_sr_check"] = res["cum_sr_check"].fillna(method="ffill").fillna(0)
        res["group_number"] = res["exist_group_cum"] + res["cum_sr_check"]

        repl = res.groupby("group_number").agg({"value": list}).to_dict()["value"]
        repl = {k: "_".join(v) for k, v in repl.items()}
        res["group_vals"] = res["group_number"].replace(repl)

        t = dict(zip(res["value"], res["group_vals"]))

        return t

    def _plot_single_woe_grouping(self, stats, ax_pd=None):
        """
        Построение графика по группировке предиктора

        Parameters
        ---------------
        stats : pandas.DataFrame
                Статистика по каждой группе (результат работы функции statistic):
                минимальное, максимальное значение, доля от общего объема выборки,
                количество и доля целевых и нецелевых событий в каждой группе,
                WOE и IV каждой группы
                Должен содержать столбцы: 'sample_rate', 'target_rate', 'WOE'
        ax_pd : matplotlib.Axes
                Набор осей (subplot)

        """
        # Расчеты
        x2 = [stats["sample_rate"][:i].sum() for i in range(stats.shape[0])] + [
            1
        ]  # доля выборки с накоплением
        x = [
            np.mean(x2[i : i + 2]) for i in range(len(x2) - 1)
        ]  # средняя точка в группах

        # Выделение нужной информации для компактности
        woe = list(stats["WOE"])
        height = list(stats["target_rate"])  # проблемность в группе
        width = list(stats["sample_rate"])  # доля выборки на группу

        # Визуализация
        if ax_pd is None:
            _, ax_pd = plt.subplots(figsize=(8, 5))

        # Столбчатая диаграмма доли целевых в группах
        ax_pd.bar(
            x=x,
            height=height,
            width=width,
            color=[0, 122 / 255, 123 / 255],
            label="Группировка",
            alpha=0.7,
        )

        # График значений WOE по группам
        ax_woe = ax_pd.twinx()  # дубликат осей координат
        ax_woe.plot(
            x, woe, lw=2, color=[37 / 255, 40 / 255, 43 / 255], label="woe", marker="o"
        )

        # Линия нулевого значения WOE
        ax_woe.plot(
            [0, 1], [0, 0], lw=1, color=[37 / 255, 40 / 255, 43 / 255], linestyle="--"
        )

        # Настройка осей координат
        plt.xlim([0, 1])
        plt.xticks(x2, [round(i, 2) for i in x2], fontsize=12)
        ax_pd.grid(True)
        ax_pd.set_xlabel("Доля выборки", fontsize=16)
        ax_pd.set_ylabel("pd", fontsize=16)
        ax_woe.set_ylabel("woe", fontsize=16)

        # Расчет границ графика и шага сетки
        max_woe = max([int(abs(i)) + 1 for i in woe])
        max_pd = max([int(i * 10) + 1 for i in height]) / 10

        # Границы и сетка для столбчатой диаграммы
        ax_pd.set_ylim([0, max_pd])
        ax_pd.set_yticks([round(i, 2) for i in np.linspace(0, max_pd, 11)])
        ax_pd.legend(bbox_to_anchor=(1.05, 0.83), loc=[0.2, -0.25], fontsize=14)

        # Границы и сетка для графика WOE
        ax_woe.set_ylim([-max_woe, max_woe])
        ax_woe.set_yticks([round(i, 2) for i in np.linspace(-max_woe, max_woe, 11)])
        ax_woe.legend(bbox_to_anchor=(1.05, 0.92), loc=[0.6, -0.25], fontsize=14)

        plt.title(
            "Группировка предиктора {}".format(stats.loc[0, "predictor"]), fontsize=18
        )

        # Для категориальных
        n_cat = stats.loc[stats["type"] == "cat"].shape[0]

        if n_cat > 0:
            ax_pd.bar(
                x=x[-n_cat:],
                height=height[-n_cat:],
                width=width[-n_cat:],
                color="m",
                label="Категориальные",
            )
            ax_pd.legend(bbox_to_anchor=(1.05, 0.76), loc=[0.15, -0.33], fontsize=14)

        plt.tight_layout()

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
            DF_i1 = DF_i1.loc[
                (DF_i1["sample_rate"] > self.min_sample_rate)
                & (~DF_i1["value"].isin(cat_vals))
            ]

            # Выделение всех значений предиктора, не отмеченных, как категориальные
            DF_i2 = self.grouped.get_predictor(i).copy()
            DF_i2 = DF_i2.loc[(~DF_i2["value"].isin(cat_vals))]

            # Выбор значений: которые не равны бесконености и при этом не являются числами
            L = ~(DF_i2["value"] == np.inf) & (~(self._get_nums_mask(DF_i2["value"])))
            DF_i2 = DF_i2.loc[L]
            # Объединение найденных значений в одну таблицу
            DF_i = pd.concat((DF_i1, DF_i2), ignore_index=True).drop_duplicates()

            self.possible_groups = self.possible_groups.append(DF_i)

    def _get_bad_groups(self):
        """
        Поиск групп: не удовлетворяющих условиям
        """
        self.bad_groups = self.stats.loc[
            (self.stats["sample_rate"] < self.min_sample_rate)
            | (self.stats["target_count"] < self.min_count)
            | (self.stats["sample_count"] - self.stats["target_count"] < self.min_count)
        ]

    def _regularize_groups(self, stats, alpha=0):
        """расчет оптимальной целевой для группы на основе готовой woe-группировки
        формула и детали в видео
        https://www.youtube.com/watch?v=g335THJxkto&list=PLLIunAIxCvT8ZYpC6-X7H0QfAQO9H0f-8&index=12&t=0s
        pd = (y_local * K + Y_global * alpha) / (K + alpha)"""
        Y_global = stats["target_count"].sum() / stats["sample_count"].sum()
        K = stats["sample_count"] / stats["sample_count"].sum()
        stats["target_rate"] = (stats["target_rate"] * K + Y_global * alpha) / (
            K + alpha
        )
        stats["target_count"] = np.floor(
            stats["sample_count"] * stats["target_rate"]
        ).astype(int)

        return stats

    def _statistic(self, grouped, alpha=0):
        """
        Расчет статистики по группам предиктора: минимальное, максимальное значение, доля от
        общего объема выборки, количество и доля целевых и нецелевых событий в каждой группе
        А также расчет WOE и IV каждой группы

        Parameters
        ---------------
        grouped : pandas.DataFrame
                Данные полученных групп предиктора. Кол-во строк совпадает с кол-вом
                уникальных значений предиктора.
                Должен содержать столбцы: 'sample_count', 'target_count', 'groups'
        alpha : float, default 0
                Коэффициент регуляризации групп

        Returns
        ---------------
        stats : pandas.DataFrame
                Агрегированные данные по каждой группе

        """
        nothing = 10 ** -6
        stats = grouped.groupby(["predictor", "groups"], sort=False).agg(
            {
                "type": "first",
                "sample_count": "sum",
                "target_count": "sum",
                "value": ["min", "max"],
            },
        )
        stats.columns = ["type", "sample_count", "target_count", "min", "max"]
        stats.reset_index(inplace=True)
        stats["sample_rate"] = stats["sample_count"] / stats["sample_count"].sum()
        stats["target_rate"] = stats["target_count"] / stats["sample_count"]

        stats = self._regularize_groups(stats, alpha=alpha)

        # Расчет WoE и IV
        samples_num = stats["sample_count"].sum()
        events = stats["target_count"].sum()
        non_events = samples_num - events

        stats["non_events_i"] = stats["sample_count"] - stats["target_count"]
        stats["event_rate_i"] = stats["target_count"] / (events + nothing)
        stats["non_event_rate_i"] = stats["non_events_i"] / (non_events + nothing)

        stats["WOE"] = np.log(
            stats["non_event_rate_i"] / (stats["event_rate_i"] + nothing) + nothing
        )

        stats["IV"] = stats["WOE"] * (stats["non_event_rate_i"] - stats["event_rate_i"])

        return stats

    def _calc_trend_coefs(self, x, y):
        """
        Расчет коэффициентов тренда

        Parameters
        ---------------
        x : pandas.Series
                Значения предиктора
        y : pandas.Series
                Целевая переменная
        Returns
        -----------
        dict[str, tuple[float, float]]
        """
        return {x.name: np.polyfit(x, y, deg=1)}

    # Служебные функции
    def _reset_state(self):
        self.trend_coefs = {}
        self.borders = {}
        self.cat_values = {}
        self.predictors = []
        self.grouped = _GroupedPredictor()
        self.stats = _GroupedPredictor()

    def _get_nums_mask(self, x):
        # if x.apply(lambda x: isinstance(x, str)).sum() == len(x):
        #     return pd.Series(False, index=x.index)
        # else:
        #     mask = pd.to_numeric(x, errors="coerce").notna()
        mask = pd.to_numeric(x, errors="coerce").notna()

        return mask


class WoeTransformerRegularized(WoeTransformer):
    """
    Класс для построения и применения WOE группировки к датасету с применением
    регуляризации малых групп
    """

    def __init__(self, min_sample_rate=0.05, min_count=3, alphas=None, n_seeds=30):
        """
        Инициализация экземпляра класса

        """
        self.min_sample_rate = min_sample_rate
        self.min_count = min_count
        self.predictors = []
        self.alphas = 100 if alphas is None else alphas
        self.alpha_values = {}
        self.n_seeds = n_seeds

    def fit(self, X, y, cat_values={}, alpha_values={}):
        """
        Обучение трансформера и расчет всех промежуточных данных

        Parameters
        ---------------
        X : pd.DataFrame
                Датафрейм с предикторами, которые нужно сгруппировать
        y : pd.Series
                Целевая переменная
        cat_values : dict[str, list[str]], optional
                Словарь списков с особыми значениями, которые нужно
                выделить в категории
                По умолчанию все строковые и пропущенные значения
                выделяются в отдельные категории
        alpha_values : dict[str, float], optional
                Словарь со значениями alpha для регуляризации WOE-групп

        Returns
        -------
        self : WoeTransformer
        """
        # Сброс текущего состояния трансформера
        self._reset_state()
        self.cat_values = cat_values
        self.regularization_stats = _GroupedPredictor()

        for col in tqdm(X.columns, desc="Searching alphas"):
            temp_alpha = self._cat_features_alpha_logloss(
                X[col], y, self.alphas, self.n_seeds
            )
            self.alpha_values.update({col: temp_alpha})

        self._grouping(X, y)
        # Расчет WOE и IV
        self._fit_numeric(X, y)
        # Поиск потенциальных групп
        # Поиск "плохих" групп
        self._get_bad_groups()

        return self

    def _cat_features_alpha_logloss(self, x, y, alphas, seed=30):
        """
        функция расчета IV, GINI и logloss для категориальных
        переменных с корректировкой целевой по alpha

        """
        # задаем промежуточную функцию для WOE преобразования переменной из исходного датафрейма
        # по рассчитанным WOE из IVWOE
        def calc_woe_i(row_value, stats):
            return stats.loc[stats["groups"] == row_value, "WOE"].values[0]

        predictor = x.name
        target = y.name
        df = pd.DataFrame({predictor: x.values, target: y.values})
        df[predictor] = df[predictor].fillna("NO_INFO")
        L_logloss_mean = []
        GINI_IV_mean = []
        for alpha_i in alphas:
            logloss_i = []
            GINI_i = []
            IV_i = []
            for seed_i in range(seed):
                X_train, X_test, y_train, y_test = train_test_split(
                    x, y, test_size=0.3, random_state=seed_i, stratify=y
                )
                # Группировка значений предиктора с текущим alpha
                df_i = self._group_single(X_train, y_train)
                df_i["groups"] = df_i["value"].fillna("пусто")
                df_i["type"] = "cat"
                # Обучение и применение группировки к обучающему набору
                WOE_i = self._fit_single(X_train, y_train, df_i)

                WOE_i = self._regularize_groups(WOE_i, alpha_i)
                # расчет оптимальной целевой для группы, формула и детали в видео
                # https://www.youtube.com/watch?v=g335THJxkto&list=PLLIunAIxCvT8ZYpC6-X7H0QfAQO9H0f-8&index=12&t=0s
                # pd = (y_local * K + Y_global * alpha) / (K + alpha)
                Y_global = y_train.mean()
                K = WOE_i["sample_count"] / WOE_i["sample_count"].sum()
                WOE_i["target_rate"] = (
                    WOE_i["target_rate"] * K + Y_global * alpha_i
                ) / (K + alpha_i)
                WOE_i["target_count"] = np.floor(
                    WOE_i["sample_count"] * WOE_i["target_rate"]
                ).astype(int)

                X_test_WOE = self._transform_single(X_test, WOE_i)

                roc_auc_i = sk.metrics.roc_auc_score(y_test, X_test_WOE)
                # Подстановка регуляризованной доли целевой вместо каждой группы
                target_transformed = X_test_WOE.replace(
                    dict(zip(WOE_i["WOE"], WOE_i["target_rate"]))
                )
                # Запись значений
                logloss_i.append(
                    sk.metrics.log_loss(y_test, target_transformed.fillna(0))
                )
                IV_i.append(WOE_i["IV"].sum())
                GINI_i.append(abs(2 * roc_auc_i - 1))
            # Запись средних значений
            L_logloss_mean.append([alpha_i, np.mean(logloss_i)])
            GINI_IV_mean.append([alpha_i, np.mean(GINI_i), np.mean(IV_i)])

        alpha_GINI_IV = pd.DataFrame(GINI_IV_mean, columns=["alpha", "GINI", "IV"])
        alpha_GINI_IV.insert(0, "predictor", predictor)
        self.regularization_stats = self.regularization_stats.append(alpha_GINI_IV)

        # Индекс значения alpha с наименьшим логлоссом
        min_logloss_ind = np.argmin(L_logloss_mean, axis=0)[1]
        alpha_opt = L_logloss_mean[min_logloss_ind][0]

        return alpha_opt


########################
# Комплект ускоренных версий функции           #
########################
# Сильно отстал от класса, но в точности повторяет функциональность Vanilla


def grouping(DF_data_i, low_acc=False):
    """
    Агрегация данных по значениям предиктора. Рассчитывает количество наблюдений,
    количество целевых событий, долю группы от общего числа наблюдений и долю целевых в группе

    Parameters
    ---------------
        DF_data_i : pandas.DataFrame
                Таблица данных для агрегации, должна содержать поля 'predictor' и 'target'.
                Поле target при этом должно состоять из 0 и 1, где 1 - целевое событие
        low_acc : int, default None
                Параметр для округления значений предиктора.
                Если None, то предиктор не округляется.
                Если целое неотрицательное число, параметр используется для определения
                количества знаков после запятой, остальные значения игнорируются

    Returns
    ---------------
        DF_grouping : pandas.DataFrame
                Таблица с агрегированными данными по значениям предиктора

    """
    # Округение, если аргумент принимает допустимые значения
    if low_acc and type(low_acc) is int and low_acc > 0:
        DF_data_i = DF_data_i[["predictor", "target"]].round(low_acc)

    # Группировка и расчет показателей
    DF_grouping = (
        DF_data_i.groupby("predictor")["target"].agg(["count", "sum"]).reset_index()
    )
    DF_grouping.columns = ["predictor", "sample_count", "target_count"]
    DF_grouping["sample_rate"] = (
        DF_grouping["sample_count"] / DF_grouping["sample_count"].sum()
    )
    DF_grouping["target_rate"] = (
        DF_grouping["target_count"] / DF_grouping["sample_count"]
    )

    return DF_grouping


def monotonic_borders(DF_grouping, p, min_sample_rate=0.05, min_count=3):
    """
    Определение оптимальных границ групп предиктора (монотонный тренд)

    Parameters
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

    Returns
    ---------------
        R_borders : list
             Правые границы групп для последующей группировки

    """
    k01, k11 = (1, 1) if p[0] > 0 else (0, -1)
    R_borders = []
    min_ind = 0  # минимальный индекс. Начальные условия

    while min_ind < DF_grouping.shape[0]:  # цикл по новым группам

        # Расчет показателей накопительным итогом
        DF_j = DF_grouping.loc[min_ind:]
        DF_iter = DF_j[["sample_rate", "sample_count", "target_count"]].cumsum()
        DF_iter["non_target_count"] = DF_iter["sample_count"] - DF_iter["target_count"]
        DF_iter["target_rate"] = DF_iter["target_count"] / DF_iter["sample_count"]

        # Проверка на соответствие критериям групп
        DF_iter["check"] = (
            (DF_iter["sample_rate"] >= min_sample_rate - 10 ** -9)
            & (DF_iter["target_count"] >= min_count)
            & (DF_iter["non_target_count"] >= min_count)
        )

        # Расчет базы для проверки оптимальности границы
        # В зависимости от тренда считается скользящий _вперед_ минимум или максимум
        # (в расчете участвуют все наблюдения от текущего до последнего)
        if k11 == 1:
            DF_iter["pd_gr"] = (
                DF_iter["target_rate"][::-1]
                .rolling(len(DF_iter), min_periods=0)
                .min()[::-1]
            )
        else:
            DF_iter["pd_gr"] = (
                DF_iter["target_rate"][::-1]
                .rolling(len(DF_iter), min_periods=0)
                .max()[::-1]
            )

        # Проверка оптимальности границы
        DF_iter["opt"] = DF_iter["target_rate"] == DF_iter["pd_gr"]
        DF_iter = pd.concat([DF_j[["predictor"]], DF_iter], axis=1)
        try:
            min_ind = DF_iter.loc[
                (DF_iter["check"]) & (DF_iter["opt"]), "target_rate"
            ].index.values[0]
            score_j = DF_iter.loc[min_ind, "predictor"]
            if (
                len(R_borders) > 0 and score_j == R_borders[-1]
            ):  # Выход из цикла, если нет оптимальных границ
                break
        except Exception:
            break
        min_ind += 1
        R_borders.append(score_j)

    # Проверка последней добавленной группы
    DF_iter = DF_grouping.loc[DF_grouping["predictor"] > R_borders[-1]]
    sample_rate_i = DF_iter["sample_rate"].sum()  # доля выборки
    sample_count_i = DF_iter["sample_count"].sum()  # количество наблюдений
    target_count_i = DF_iter["target_count"].sum()  # количество целевых
    non_target_count_i = sample_count_i - target_count_i  # количество нецелевых

    if (
        (sample_rate_i < min_sample_rate)
        or (target_count_i < min_count)
        or (non_target_count_i < min_count)
    ):
        R_borders.remove(R_borders[-1])  # удаление последней границы
    return R_borders


# Статистика
def statistic(DF_groups):
    """
    Расчет статистики по группам предиктора: минимальное, максимальное значение, доля от
    общего объема выборки, количество и доля целевых и нецелевых событий в каждой группе
    А также расчет WOE и IV каждой группы

    Parameters
    ---------------
        DF_groups : pandas.DataFrame
                Данные полученных групп предиктора. Кол-во строк совпадает с кол-вом
                уникальных значений предиктора.
                Должен содержать столбцы: 'sample_count', 'target_count', 'groups'

    Returns
    ---------------
        DF_statistic : pandas.DataFrame
                Агрегированные данные по каждой группе

    """

    nothing = 10 ** -6
    DF_statistic = (
        DF_groups[["sample_count", "target_count", "groups"]]
        .groupby("groups", as_index=False, sort=False)
        .sum()
    )
    DF_statistic_min = (
        DF_groups[["predictor", "groups"]]
        .groupby("groups", as_index=False, sort=False)
        .min()
    )
    DF_statistic_max = (
        DF_groups[["predictor", "groups"]]
        .groupby("groups", as_index=False, sort=False)
        .max()
    )
    DF_statistic["min"] = DF_statistic_min["predictor"]
    DF_statistic["max"] = DF_statistic_max["predictor"]
    DF_statistic["sample_rate"] = (
        DF_statistic["sample_count"] / DF_statistic["sample_count"].sum()
    )
    DF_statistic["target_rate"] = (
        DF_statistic["target_count"] / DF_statistic["sample_count"]
    )

    # Расчет WoE и IV
    samples_num = DF_statistic["sample_count"].sum()
    events = DF_statistic["target_count"].sum()
    non_events = samples_num - events

    DF_statistic["non_events_i"] = (
        DF_statistic["sample_count"] - DF_statistic["target_count"]
    )
    DF_statistic["event_rate_i"] = DF_statistic["target_count"] / (events + nothing)
    DF_statistic["non_event_rate_i"] = DF_statistic["non_events_i"] / (
        non_events + nothing
    )

    DF_statistic["WOE"] = np.log(
        DF_statistic["non_event_rate_i"] / (DF_statistic["event_rate_i"] + nothing)
        + nothing
    )

    DF_statistic["IV"] = DF_statistic["WOE"] * (
        DF_statistic["non_event_rate_i"] - DF_statistic["event_rate_i"]
    )

    DF_statistic = DF_statistic.merge(
        DF_groups[["type", "groups"]].drop_duplicates(), how="left", on="groups"
    )

    return DF_statistic


# Графики
def group_plot(DF_result):
    """
    Построение графика по группировке предиктора

    Parameters
    ---------------
        DF_result : pandas.DataFrame
                Статистика по каждой группе (результат работы функции statistic):
                минимальное, максимальное значение, доля от общего объема выборки,
                количество и доля целевых и нецелевых событий в каждой группе,
                WOE и IV каждой группы
                Должен содержать столбцы: 'sample_rate', 'target_rate', 'WOE'

    Returns
    ---------------
        None
                Не возвращает ничего

    """

    # Расчеты
    sample_rate, target_rate, WOE = ["sample_rate", "target_rate", "WOE"]

    x2 = [DF_result[sample_rate][:i].sum() for i in range(DF_result.shape[0])] + [
        1
    ]  # доля выборки с накоплением
    x = [np.mean(x2[i : i + 2]) for i in range(len(x2) - 1)]  # средняя точка в группах

    # Выделение нужной информации для компактности
    woe = list(DF_result[WOE])
    height = list(DF_result[target_rate])  # проблемность в группе
    width = list(DF_result[sample_rate])  # доля выборки на группу

    # Визуализация
    fig, ax_pd = plt.subplots(figsize=(8, 5))

    # Столбчатая диаграмма доли целевых в группах
    ax_pd.bar(
        x=x,
        height=height,
        width=width,
        color=[0, 122 / 255, 123 / 255],
        label="Группировка",
        alpha=0.7,
    )

    # График значений WOE по группам
    ax_woe = ax_pd.twinx()  # дубликат осей координат
    ax_woe.plot(
        x, woe, lw=2, color=[37 / 255, 40 / 255, 43 / 255], label="woe", marker="o"
    )

    # Линия нулевого значения WOE
    ax_woe.plot(
        [0, 1], [0, 0], lw=1, color=[37 / 255, 40 / 255, 43 / 255], linestyle="--"
    )

    # Настройка осей координат
    plt.xlim([0, 1])
    plt.xticks(x2, [round(i, 2) for i in x2], fontsize=12)
    ax_pd.grid(True)
    ax_pd.set_xlabel("Доля выборки", fontsize=16)
    ax_pd.set_ylabel("pd", fontsize=16)
    ax_woe.set_ylabel("woe", fontsize=16)

    # Расчет границ графика и шага сетки
    max_woe = max([int(abs(i)) + 1 for i in woe])
    max_pd = max([int(i * 10) + 1 for i in height]) / 10
    # Границы и сетка для столбчатой диаграммы
    ax_pd.set_ylim([0, max_pd])
    ax_pd.set_yticks([round(i, 2) for i in np.linspace(0, max_pd, 11)])
    ax_pd.legend(loc=[0.2, -0.25], fontsize=14)
    # Границы и сетка для графика WOE
    ax_woe.set_ylim([-max_woe, max_woe])
    ax_woe.set_yticks([round(i, 2) for i in np.linspace(-max_woe, max_woe, 11)])
    ax_woe.legend(loc=[0.6, -0.25], fontsize=14)

    plt.title("Группировка предиктора", fontsize=18)

    # Для категориальных
    n_cat = DF_result.loc[DF_result["type"] == "cat"].shape[0]

    if n_cat > 0:
        ax_pd.bar(
            x=x[-n_cat:],
            height=height[-n_cat:],
            width=width[-n_cat:],
            color="m",
            label="Категориальные",
        )
        ax_pd.legend(loc=[0.15, -0.33], fontsize=14)

    plt.show()


# ## Трансформер
def woe_transformer(
    x,
    y,
    cat_values=[],
    min_sample_rate=0.05,
    min_count=3,
    errors="skip",
    low_accuracy=None,
    plot=True,
    verbose=True,
):
    """
    Группировка значений предиктора, определение оптимальных границ и расчет WOE и IV

    Parameters
    ---------------
        x : pandas.Series
                Mассив числовых значений предиктора. Не должен содержать пропущенных
                значений, но может сочетать строковые и числовые
        y : pandas.Series
                Mассив меток класса (0, 1)
        cat_values: list
                Категориальные значения (пустышки и несравнимые значения).
                Элементы списка должны быть строками
        min_sample_rate : float, default 0.05
                Минимальный размер группы (доля от размера выборки)
        min_count : int, default 3
                Минимальное количество наблюдений каждого класса в группе
        errors : str, defaulf 'skip'
                Способ обработки ошибок:
                    'skip' - не возвращать ничего в случае ошибки
                    'origin' - вернуть исходные значения предиктора
                    'raise' - бросить исключение
        low_accuracy : int, default None
                Режим пониженной точности (округление при группировке)
                Если None, то предиктор не округляется.
                Если целое неотрицательное число, параметр используется для определения
                количества знаков после запятой, остальные значения игнорируются
        plot : bool, default True
                Включение/выключение визуализации группировки
        verbose : bool, default True
                        Включение.выключение доп. информации по группировке

    Returns
    ---------------
        DF_result : pandas.DataFrame
                Таблица с итоговой группировкой и статистикой

    """
    if errors not in ["skip", "raise"]:
        warnings.warn(
            f"Attribute `errors` must be one of ['skip', 'raise']. Passed {errors}.\n\
                Defaulting to 'skip'"
        )
        errors = "skip"

    # Обработка входных данных
    DF_data_i = pd.DataFrame({"predictor": x, "target": y})

    # Агрегация данных по значениям предиктора
    DF_data_gr = grouping(DF_data_i, low_accuracy)

    # Проверка категориальных групп (возможные дополнительные категории)
    if verbose:
        # Выделение значений предиктора с достаточным кол-вом наблюдений и
        # не отмеченных, как категориальные
        DF_i1 = DF_data_gr.loc[DF_data_gr["sample_rate"] > min_sample_rate].loc[
            ~DF_data_gr["predictor"].isin(cat_values)
        ]

        # Выделение всех значений предиктора, не отмеченных, как категориальные
        DF_i2 = DF_data_gr.loc[~DF_data_gr["predictor"].isin(cat_values)]

        # Выбор значений: которые не равны бесконености и при этом не являются числами
        L = ~(DF_i2["predictor"] == np.inf) & (
            pd.to_numeric(DF_i2["predictor"], errors="coerce").isna()
        )

        DF_i2 = DF_i2.loc[L]
        # Объединение найденных значений в одну таблицу
        DF_i = DF_i1.append(DF_i2, ignore_index=True).drop_duplicates()
        if DF_i.shape[0] > 0:
            print("Возможно эти значения предиктора тоже являются категориальными:")
            display(DF_i)

    # Выделение числовых значений предиктора
    DF_data_gr_num = DF_data_gr.loc[
        ~DF_data_gr["predictor"].isin(cat_values)
    ].reset_index(drop=True)

    if DF_data_gr_num.shape[0] > 0:
        try:
            DF_data_gr_num["predictor"] = DF_data_gr_num["predictor"].astype("float")

            # Определение тренда по числовым значениям
            DF_i = DF_data_i.loc[~DF_data_i["predictor"].isin(cat_values)]
            p = np.polyfit(DF_i["predictor"].astype("float"), DF_i["target"], deg=1)
            # Определение оптимальных границ групп
            R_borders = monotonic_borders(DF_data_gr_num, p, min_sample_rate, min_count)
        except Exception:
            if errors == "raise":
                raise ValueError("Ошибка при расчете монотонных границ")
            else:
                print("Ошибка при расчете монотонных границ")

        try:
            # Применение границ
            DF_data_gr_num["groups"] = pd.cut(
                DF_data_gr_num["predictor"], [-np.inf] + R_borders + [np.inf]
            )
            DF_data_gr_num["type"] = "num"
        except Exception:
            if errors == "raise":
                raise ValueError("Ошибка при применении монотонных границ")
            else:
                print("Ошибка при применении монотонных границ")

    # Добавление данных по категориальным значениям
    DF_data_gr_2k = DF_data_gr.loc[
        DF_data_gr["predictor"].isin(cat_values)
    ].reset_index(drop=True)
    DF_data_gr_2k["groups"] = DF_data_gr_2k["predictor"].copy()
    DF_data_gr_2k["type"] = "cat"

    try:
        # Расчет статистики, WoE и IV по группам числовых значений
        if DF_data_gr_num.shape[0] > 0:
            DF_result = statistic(
                DF_data_gr_num.append(DF_data_gr_2k, ignore_index=True)
            )
        else:
            DF_result = statistic(DF_data_gr_2k)
    except Exception:
        print("Ошибка при расчете статистики")

    # Проверка категориальных групп (категории, которые не удовлетворяют заданным ограничениям)
    if verbose:
        DF_j = DF_result.loc[
            (DF_result["sample_rate"] < min_sample_rate)
            | (DF_result["target_count"] < min_count)
            | (DF_result["sample_count"] - DF_result["target_count"] < min_count)
        ]
        if DF_j.shape[0] > 0:
            print("Эти группы не удовлетворяют заданным ограничениям:")
            display(DF_j)
        # Построение графика
        if plot:
            group_plot(DF_result)

    return DF_result


def woe_apply(S_data, DF_groups):
    """
    Применение группировки и WoE-преобразования

    Parameters---------------
        S_data : pandas.Series
                Значения предиктора
        DF_groups : pandas.DataFrame
                Данные о группировке предиктора

    Returns
    ---------------
        X_woe : pandas.DataFrame
                WoE-преобразования значений предиктора
                WoE = 0, если группа не встречалась в обучающей выборке

    """
    X_woe = S_data.copy()
    # Маппинги для замены групп на соответствующие значения WOE
    num_map = {
        DF_groups.loc[i, "groups"]: DF_groups.loc[i, "WOE"]
        for i in DF_groups.index
        if DF_groups.loc[i, "type"] == "num"
    }
    cat_map = {
        DF_groups.loc[i, "groups"]: DF_groups.loc[i, "WOE"]
        for i in DF_groups.index
        if DF_groups.loc[i, "type"] == "cat"
    }
    # Категориальные группы
    cat_bounds = DF_groups.loc[DF_groups["type"] == "cat", "groups"]

    # predict по числовым значениям
    DF_num = DF_groups.loc[DF_groups["type"] == "num"]
    if DF_num.shape[0] > 0:
        # Границы (правые) интервалов для разбивки числовых переменных
        num_bounds = [-np.inf] + list(
            pd.IntervalIndex(DF_groups.loc[DF_groups["type"] == "num", "groups"]).right
        )
        # Выделение только числовых значений предиктора
        # (похожих на числа и тех, что явно не указаны как категориальные)
        X_woe_num = X_woe[
            X_woe.astype(str)
            .str.replace(r"\.|\-", "")
            .str.replace("e", "")
            .str.isdecimal()
            & (~X_woe.isin(cat_bounds))
        ]
        # Разбивка значений на интервалы в соответствии с группировкой
        X_woe_num = pd.cut(X_woe_num, num_bounds)
        # Замена групп на значения WOE
        X_woe_num = X_woe_num.replace(num_map)
        X_woe_num.name = "woe"
    else:
        X_woe_num = pd.Series()

    # predict по категориальным значениям (может обновлять значения по числовым)
    DF_cat = DF_groups.loc[DF_groups["type"] == "cat"]
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


########################
# Комплект Vanilla-версий функции              #
########################


def _grouping(DF_data_i):
    """
    Агрегация данных по значениям предиктора
    DF_data_i[['predictor', 'target']] - таблица данных
    """
    DF_i = (
        DF_data_i[["predictor", "target"]].groupby("predictor", as_index=False).count()
    )
    DF_j = DF_data_i[["predictor", "target"]].groupby("predictor", as_index=False).sum()
    DF_grouping = DF_i.merge(DF_j, how="left", on="predictor")
    DF_grouping.columns = ["predictor", "sample_count", "target_count"]
    DF_grouping["sample_rate"] = (
        DF_grouping["sample_count"] / DF_grouping["sample_count"].sum()
    )
    DF_grouping["target_rate"] = (
        DF_grouping["target_count"] / DF_grouping["sample_count"]
    )

    return DF_grouping


def _monotonic_borders(DF_grouping, p, min_sample_rate=0.05, min_count=3):
    """
    Vanilla-версия функции, оставлена на всякий случай

    Определение оптимальных границ групп (монотонный тренд)
    DF_grouping - агрегированные данные по значениям предиктора
    DF_grouping[['predictor', 'sample_count', 'target_count', 'sample_rate', 'target_rate]]
    min_sample_rate - минимальный размер группы (доля от размера выборки)
    min_count - минимальное количество наблюдений каждого класса в группе
    """
    k01, k11 = (1, 1) if p[0] > 0 else (0, -1)
    L_borders = []
    min_ind = 0  # минимальный индекс. Начальные условия

    while min_ind < DF_grouping.shape[0]:  # цикл по новым группам
        pd_gr_i = (
            k01  # средняя pd в группе. Начальные условия (зависит от общего тренда)
        )

        for j in range(min_ind, max(DF_grouping.index) + 1):  # цикл по конечной границе
            DF_j = DF_grouping.loc[min_ind:j]
            sample_rate_i = DF_j["sample_rate"].sum()  # доля выборки
            sample_count_i = DF_j["sample_count"].sum()  # количество наблюдений
            target_count_i = DF_j["target_count"].sum()  # количество целевых
            non_target_count_i = sample_count_i - target_count_i  # количество нецелевых
            target_rate_i = target_count_i / sample_count_i

            if (
                (sample_rate_i < min_sample_rate)
                or (target_count_i < min_count)
                or (non_target_count_i < min_count)
            ):
                continue  # если граница не удовлетворяет условиям

            if target_rate_i * k11 < pd_gr_i * k11:  # проверка оптимальности границы
                min_ind_i = j + 1
                pd_gr_i = target_rate_i
                score_j = DF_grouping.loc[j, "predictor"]

        min_ind = min_ind_i
        if (
            len(L_borders) > 0 and score_j == L_borders[-1]
        ):  # Выход из цикла, если нет оптимальных границ
            break
        L_borders.append(score_j)

    # Проверка последней добавленной группы

    DF_j = DF_grouping.loc[DF_grouping["predictor"] > L_borders[-1]]
    sample_rate_i = DF_j["sample_rate"].sum()  # доля выборки
    sample_count_i = DF_j["sample_count"].sum()  # количество наблюдений
    target_count_i = DF_j["target_count"].sum()  # количество целевых
    non_target_count_i = sample_count_i - target_count_i  # количество нецелевых

    if (
        (sample_rate_i < min_sample_rate)
        or (target_count_i < min_count)
        or (non_target_count_i < min_count)
    ):
        L_borders.remove(L_borders[-1])  # удаление последней границы

    return L_borders


def _statistic(DF_groups):
    """
    Vanilla-версия функции, оставлена на всякий случай

    Расчет статистики по группам
    DF_groups[['sample_count', 'target_count', 'groups']] - таблица данных по группам
    """
    nothing = 10 ** -6
    DF_statistic = (
        DF_groups[["sample_count", "target_count", "groups"]]
        .groupby("groups", as_index=False, sort=False)
        .sum()
    )
    DF_statistic_min = (
        DF_groups[["predictor", "groups"]]
        .groupby("groups", as_index=False, sort=False)
        .min()
    )
    DF_statistic_max = (
        DF_groups[["predictor", "groups"]]
        .groupby("groups", as_index=False, sort=False)
        .max()
    )
    DF_statistic["min"] = DF_statistic_min["predictor"]
    DF_statistic["max"] = DF_statistic_max["predictor"]
    DF_statistic["sample_rate"] = (
        DF_statistic["sample_count"] / DF_statistic["sample_count"].sum()
    )
    DF_statistic["target_rate"] = (
        DF_statistic["target_count"] / DF_statistic["sample_count"]
    )

    # Расчет WoE и IV
    samples_num = DF_statistic["sample_count"].sum()
    events = DF_statistic["target_count"].sum()
    non_events = samples_num - events

    DF_statistic["non_events_i"] = (
        DF_statistic["sample_count"] - DF_statistic["target_count"]
    )
    DF_statistic["event_rate_i"] = DF_statistic["target_count"] / (events + nothing)
    DF_statistic["non_event_rate_i"] = DF_statistic["non_events_i"] / (
        non_events + nothing
    )

    DF_statistic["WOE"] = [
        math.log(
            DF_statistic["non_event_rate_i"][i]
            / (DF_statistic["event_rate_i"][i] + nothing)
            + nothing
        )
        for i in DF_statistic.index
    ]
    DF_statistic["IV"] = DF_statistic["WOE"] * (
        DF_statistic["non_event_rate_i"] - DF_statistic["event_rate_i"]
    )

    DF_statistic = DF_statistic.merge(
        DF_groups[["type", "groups"]].drop_duplicates(), how="left", on="groups"
    )

    return DF_statistic


def _group_plot(DF_result, L_cols=["sample_rate", "target_rate", "WOE"]):
    """
    Vanilla-версия функции, оставлена на всякий случай

    Построение графика по группировке предиктора
    DF_result - таблица данных
    L_cols - список названий столбцов
    L_cols = ['sample_rate', 'target_rate', 'WOE']
    """
    [sample_rate, target_rate, WOE] = L_cols

    fig, ax_pd = plt.subplots(figsize=(8, 5))

    x2 = [DF_result[sample_rate][:i].sum() for i in range(DF_result.shape[0])] + [
        1
    ]  # доля выборки с накоплением
    x = [np.mean(x2[i : i + 2]) for i in range(len(x2) - 1)]  # средняя точка в группах
    woe = list(DF_result[WOE])
    height = list(DF_result[target_rate])  # проблемность в группе
    width = list(DF_result[sample_rate])  # доля выборки на группу

    ax_pd.bar(
        x=x,
        height=height,
        width=width,
        color=[0, 122 / 255, 123 / 255],
        label="Группировка",
        alpha=0.7,
    )

    ax_woe = ax_pd.twinx()
    ax_woe.plot(
        x, woe, lw=2, color=[37 / 255, 40 / 255, 43 / 255], label="woe", marker="o"
    )
    ax_woe.plot(
        [0, 1], [0, 0], lw=1, color=[37 / 255, 40 / 255, 43 / 255], linestyle="--"
    )

    plt.xlim([0, 1])
    plt.xticks(x2, [round(i, 2) for i in x2], fontsize=12)
    ax_pd.grid(True)
    ax_pd.set_xlabel("Доля выборки", fontsize=16)
    ax_pd.set_ylabel("pd", fontsize=16)
    ax_woe.set_ylabel("woe", fontsize=16)

    # расчет границ графика и шага сетки
    max_woe = max([int(abs(i)) + 1 for i in woe])

    max_pd = max([int(i * 10) + 1 for i in height]) / 10

    ax_pd.set_ylim([0, max_pd])
    ax_woe.set_ylim([-max_woe, max_woe])

    ax_pd.set_yticks([round(i, 2) for i in np.linspace(0, max_pd, 11)])
    ax_woe.set_yticks([round(i, 2) for i in np.linspace(-max_woe, max_woe, 11)])

    plt.title("Группировка предиктора", fontsize=18)

    ax_pd.legend(loc=[0.2, -0.25], fontsize=14)
    ax_woe.legend(loc=[0.6, -0.25], fontsize=14)

    # для категориальных
    n_cat = DF_result.loc[DF_result["type"] == "cat"].shape[0]

    if n_cat > 0:
        ax_pd.bar(
            x=x[-n_cat:],
            height=height[-n_cat:],
            width=width[-n_cat:],
            color="m",
            label="Категориальные",
        )
        ax_pd.legend(loc=[0.15, -0.33], fontsize=14)

    plt.show()


# %% ExecuteTime={"start_time": "2020-03-25T11:06:21.844897Z", "end_time": "2020-03-25T11:06:21.855955Z"}
def _woeTransformer(
    x, y, cat_values=[], min_sample_rate=0.05, min_count=3, monotonic=True, plot=True
):
    """
    Vanilla-версия функции, оставлена на всякий случай

    woeTransformer - определяет оптимальные границы групп по заданным ограничениям
    x - массив числовых значений предиктора
    y - массив меток класса (0, 1)
    cat_values - категориальные значения (пустышки и несравнимые значения - для монотонного тренда!)
    min_sample_rate - минимальный размер группы (доля от размера выборки)
    min_count - минимальное количество наблюдений каждого класса в группе
    monotonic - монотонный тренд
    """
    # Обработка входных данных
    DF_data_i = pd.DataFrame()
    DF_data_i["predictor"] = x
    DF_data_i["target"] = y

    # Агрегация данных по значениям предиктора
    DF_data_gr = _grouping(DF_data_i)

    # Проверка категориальных групп (возможные дополнительные категории)
    # 1) возможные дополнительные категории
    DF_i1 = DF_data_gr.loc[DF_data_gr["sample_rate"] > min_sample_rate].loc[
        ~DF_data_gr["predictor"].isin(cat_values)
    ]
    DF_i2 = DF_data_gr.loc[~DF_data_gr["predictor"].isin(cat_values)]
    L = []
    for i in DF_i2["predictor"]:
        try:
            L.append(np.inf < i)
        except Exception:
            L.append(True)
    DF_i2 = DF_i2.loc[pd.Series(L, index=DF_i2.index)]
    DF_i = DF_i1.append(DF_i2, ignore_index=True).drop_duplicates()
    if DF_i.shape[0] > 0:
        print("Возможно эти значения предиктора тоже являются категориальными:")
        display(DF_i)

    try:
        # Выделение числовых значений предиктора
        DF_data_gr_2 = DF_data_gr.loc[
            ~DF_data_gr["predictor"].isin(cat_values)
        ].reset_index(drop=True)
        if DF_data_gr_2.shape[0] > 0:
            DF_data_gr_2["predictor"] = DF_data_gr_2["predictor"].astype("float")

            # Определение тренда по числовым значениям
            DF_i = DF_data_i.loc[~DF_data_i["predictor"].isin(cat_values)]
            p = np.polyfit(DF_i["predictor"].astype("float"), DF_i["target"], deg=1)

            # Определение оптимальных границ групп
            L_borders = _monotonic_borders(DF_data_gr_2, p, min_sample_rate, min_count)

            # Применение границ
            DF_data_gr_2["groups"] = pd.cut(
                DF_data_gr_2["predictor"], [-np.inf] + L_borders + [np.inf]
            )
            DF_data_gr_2["type"] = "num"

        # Добавление данных по категориальным значениям
        DF_data_gr_2k = DF_data_gr.loc[
            DF_data_gr["predictor"].isin(cat_values)
        ].reset_index(drop=True)
        DF_data_gr_2k["groups"] = DF_data_gr_2k["predictor"].copy()
        DF_data_gr_2k["type"] = "cat"

        # Расчет статистики, WoE и IV по группам числовых значений
        if DF_data_gr_2.shape[0] > 0:
            DF_result = _statistic(
                DF_data_gr_2.append(DF_data_gr_2k, ignore_index=True)
            )
        else:
            DF_result = _statistic(DF_data_gr_2k)

        # Проверка категориальных групп (категории, которые не удовлетворяют заданным ограничениям)
        DF_j = DF_result[
            (DF_result["sample_rate"] < min_sample_rate)
            | (DF_result["target_count"] < min_count)
            | (DF_result["sample_count"] - DF_result["target_count"] < min_count)
        ]
        if DF_j.shape[0] > 0:
            print("Эти группы не удовлетворяют заданным ограничениям:")
            display(DF_j)

        # Построение графика
        if plot:
            _group_plot(DF_result, ["sample_rate", "target_rate", "WOE"])

        return DF_result
    except Exception:
        print("Ошибка при выполнении группировки")


def _woe_apply(S_data, DF_groups):
    """
    Применение группировки и WoE-преобразования

    S_data - значения предиктора (pd.Series)
    DF_groups - данные о группировке
    X_woe - WoE-преобразования значений предиктора

    WoE = 0, если группа не встречалась в обучающей выборке
    """
    X_woe = S_data.copy()

    # predict по числовым значениям
    DF_num = DF_groups.loc[DF_groups["type"] == "num"]
    if DF_num.shape[0] > 0:
        for i in DF_num.index:  # цикл по группам
            group_i = DF_num["groups"][i]
            woe_i = DF_num["WOE"][i]
            values = [
                woe_i if (type(S_data[j]) != str and S_data[j] in group_i) else X_woe[j]
                for j in S_data.index
            ]
            X_woe = pd.Series(values, S_data.index, name="woe")

    # predict по категориальным значениям (может обновлять значения по числовым)
    DF_cat = DF_groups.loc[DF_groups["type"] == "cat"]
    if DF_cat.shape[0] > 0:
        for i in DF_cat.index:  # цикл по группам
            group_i = DF_cat["groups"][i]
            woe_i = DF_cat["WOE"][i]
            values = []
            for j in S_data.index:  # цикл по строкам
                try:
                    if S_data[j] == group_i:
                        values.append(woe_i)
                    else:
                        values.append(X_woe[j])
                except Exception:
                    values.append(X_woe[j])
            X_woe = pd.Series(values, S_data.index, name="woe")

    # WoE = 0, если группа не встречалась в обучающей выборке
    X_woe.loc[~X_woe.isin(DF_groups["WOE"])] = 0.0

    return X_woe
