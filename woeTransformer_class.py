"""
WOE-трансформер в виде класса с интерфейсом как в sklearn - с методами fit
и predict

TODO:
обработка серий наравне с датафреймами
_grouping - обработка случаев, когда предиктор принимает только одно значение и есть пропуски
_grouping - обработка случаев, когда предиктор принимает только одно значение и НЕТ пропусков (сейчас должно падать)
_monotonic_borders - обработка случаев: когда много пустых значений и доля непустых - меньше заданного min_sample_rate
обработка исключений и сохранения статусов обучения
обработка исключений и сохранения статусов преобразования
предупреждения о сильных различиях в распределении трейна и теста
Актуализировать и дополнить документацию
добавить регуляризацию IV

"""
import math
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.base import TransformerMixin, BaseEstimator



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

    def append(self, other):
        return _GroupedPredictor(super().append(other))


class WoeTransformer(TransformerMixin, BaseEstimator):
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

    def __repr__(self):
        return "WoeTransformer(min_sample_rate=%r, min_count=%r, n_fitted_predictors=%r)" % (self.min_sample_rate, self.min_count, len(self.predictors))

    def __init__(self, min_sample_rate=0.05, min_count=3, cat_values=None, alpha_values=None):
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
        self.cat_values = {}
        self.alpha_values = {}

        if isinstance(cat_values, dict): self.cat_values.update(cat_values)
        if isinstance(alpha_values, dict): self.alpha_values.update(alpha_values)

    # -------------------------
    # Функции интерфейса класса
    # -------------------------

    def fit(self, X, y, cat_values=None, alpha_values=None):
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
        # Сохранение категориальных знаений
        if isinstance(cat_values, dict):
            self.cat_values.update(cat_values)
        # Инициализация коэффициентов для регуляризации групп
        self.alpha_values = {i:0 for i in X.columns}
        if isinstance(alpha_values, dict):
            self.alpha_values.update(alpha_values)

        # Агрегация значений предикторов
        self._grouping(X, y)
        # Расчет WOE и IV
        self._fit_numeric(X, y)
        # Поиск потенциальных групп
        # Поиск "плохих" групп
        self._get_bad_groups()

        return self

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
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)
        for i in X:
            if i in self.predictors:
                try:
                    transformed[i] = self._transform_single(X[i])
                except Exception as e:
                    print('Transform failed on predictor: {i}'.format(i), e)
            else:
                print(f"Column is not in fitted predictors list: {i}")
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

    # -------------------------
    # Внутренние функции над всем датасетом
    # -------------------------
    def _grouping(self, X, y):
        """
        Применение группировки ко всем предикторам
        """
        df = X.copy()
        df = df.fillna('пусто')
        df['target'] = y.copy()

        # Группировка и расчет показателей
        for col in df.columns[:-1]:
            alpha = self.alpha_values.get(col, 0)
            grouped_temp = self._group_single(df[col], y, alpha)
            self.grouped = self.grouped.append(grouped_temp)

        # Замена пустых значений обратно на np.nan ИЛИ преобразование в числовой тип
        try:
            self.grouped['value'] = self.grouped['value'].replace({'пусто':np.nan})
        except TypeError:
            self.grouped['value'] = pd.to_numeric(self.grouped['value'], downcast='signed')

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

        res = pd.DataFrame()

        for i in X:
            res_i = self._fit_single(X[i], y)
            res = res.append(res_i)
            self.predictors.append(i)
        self.stats = self.stats.append(res)

    # -------------------------
    # Внутренние функции над отдельными столбцами
    # -------------------------
    def _group_single(self, x, y, alpha=0):
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
        col = x.name
        df = pd.DataFrame({col: x.values,
                           'target':y.values})
        grouped_temp = df.groupby(col)['target'].agg(['count', 'sum']).reset_index()
        grouped_temp.columns = ['value', 'sample_count', 'target_count']
        grouped_temp['sample_rate'] = grouped_temp['sample_count'] / grouped_temp['sample_count'].sum()
        grouped_temp['target_rate'] = grouped_temp['target_count'] / grouped_temp['sample_count']
        grouped_temp.insert(0, 'predictor', col)

        # расчет оптимальной целевой для группы, формула и детали в видео
        # https://www.youtube.com/watch?v=g335THJxkto&list=PLLIunAIxCvT8ZYpC6-X7H0QfAQO9H0f-8&index=12&t=0s
        # pd = (y_local * K + Y_global * alpha) / (K + alpha)
        Y_global = y.mean()
        K = grouped_temp['sample_count'] / grouped_temp['sample_count'].sum()
        grouped_temp['target_rate'] = (grouped_temp['target_rate'] * K + Y_global * alpha) / (K + alpha)
        grouped_temp['target_count'] = np.floor(grouped_temp['sample_count'] * grouped_temp['target_rate']).astype(int)

        return _GroupedPredictor(grouped_temp)

    def _fit_single(self, x, y, gr_subset=None, cat_vals=None):
        """
        Расчет WOE и IV

        Входные данные:
        ---------------
            X : pd.DataFrame
                    Датафрейм с предикторами, которые нужно сгруппировать
            y : pd.Series
                    Целевая переменная
            col : str
                    Предиктор
        """
        gr_subset_num = pd.DataFrame()
        gr_subset_cat = pd.DataFrame()
        col = x.name
        if gr_subset is None: gr_subset = (self.grouped.get_predictor(col))
        if cat_vals is None: cat_vals = self.cat_values.get(col, [])
        nan_mask = x.isna()
        num_mask = self._get_nums_mask(x) & (~x.isin(cat_vals)) & (~nan_mask)
        num_vals = x.loc[num_mask].unique()

        try:
            # Расчет коэффициентов тренда по числовым значениям предиктора
            if num_mask.sum() > 0:
                self.trend_coefs.update({col: np.polyfit(x.loc[num_mask].astype(float),
                                                    y.loc[num_mask],
                                                    deg=1)})
                # Расчет монотонных границ
                gr_subset_num = gr_subset[gr_subset['value'].isin(num_vals)].copy()
                gr_subset_num['value'] = pd.to_numeric(gr_subset_num['value'])
                borders = self._monotonic_borders(gr_subset_num,
                                                self.trend_coefs[col])
                self.borders.update({col:borders})
                # Применение границ к сгруппированным данным
                gr_subset_num['groups'] = pd.cut(gr_subset_num['value'], borders)
                gr_subset_num['type'] = 'num'
        except np.linalg.LinAlgError as e:
            print(f"Error in np.polyfit on predictor: '{col}'.\nError MSG: {e}")

        # Расчет коэффициентов тренда по категориальным значениям предиктора
        if (~num_mask).sum() > 0:
            gr_subset_cat = gr_subset[~gr_subset['value'].isin(num_vals)].copy()
            gr_subset_cat['groups'] = gr_subset_cat['value'].fillna('пусто')
            gr_subset_cat['type'] = 'cat'


        # Объединение числовых и категориальных значений
        gr_subset = pd.concat([gr_subset_num, gr_subset_cat], axis=0, ignore_index=True)

        res_i = self._statistic(gr_subset)
        is_empty_exists = any(res_i['groups'].astype(str).str.contains('пусто'))
        if is_empty_exists:
            res_i['groups'].replace({'пусто':np.nan}, inplace=True)

        return res_i

    def _transform_single(self, x, stats=None):
        """
        Применение группировки и WoE-преобразования

        Входные данные:
        ---------------
            x : pandas.Series
                    Значения предиктора
        Возвращает:
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
        num_map = {stats.loc[i, 'groups']: stats.loc[i, 'WOE']
                   for i in stats.index if stats.loc[i, 'type'] == 'num'}
        cat_map = {stats.loc[i, 'groups']: stats.loc[i, 'WOE']
                   for i in stats.index if stats.loc[i, 'type'] == 'cat'}
        # Категориальные группы
        cat_bounds = stats.loc[stats['type'] == 'cat', 'groups']

        # predict по числовым значениям
        DF_num = stats.loc[stats['type'] == 'num']
        if DF_num.shape[0] > 0:
            # Границы (правые) интервалов для разбивки числовых переменных
            num_bounds = [-np.inf] + list(pd.IntervalIndex(stats.loc[stats['type'] == 'num', 'groups']).right)
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
        DF_cat = stats.loc[stats['type'] == 'cat']
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
        X_woe = pd.to_numeric(X_woe, downcast='signed')

        return X_woe

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
        if len(R_borders) > 0:
            DF_iter = DF_grouping.loc[DF_grouping['value'] > R_borders[-1]]
            sample_rate_i = DF_iter['sample_rate'].sum()  # доля выборки
            sample_count_i = DF_iter['sample_count'].sum()  # количество наблюдений
            target_count_i = DF_iter['target_count'].sum()  # количество целевых
            non_target_count_i = sample_count_i - target_count_i  # количество нецелевых

            if (sample_rate_i < self.min_sample_rate) or (target_count_i < self.min_count) or (non_target_count_i < self.min_count):
                R_borders.remove(R_borders[-1])  # удаление последней границы
        else:
            predictor = DF_grouping['predictor'].iloc[0]
            warnings.warn("Couldn't find any borders for feature {}.\n Borders set on (-inf, +inf)".format(predictor))
        R_borders = [-np.inf] + R_borders + [np.inf]
        return R_borders

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


        plt.title('Группировка предиктора {}'.format(stats.loc[0, "predictor"]), fontsize=18)

        # Для категориальных
        n_cat = stats.loc[stats['type'] == 'cat'].shape[0]

        if n_cat > 0:
            bar_pd = ax_pd.bar(x=x[-n_cat:], height=height[-n_cat:], width=width[-n_cat:], color='m',
                               label='Категориальные')
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


    def _statistic(self, stats):
        """
        Расчет статистики по группам предиктора: минимальное, максимальное значение, доля от
        общего объема выборки, количество и доля целевых и нецелевых событий в каждой группе
        А также расчет WOE и IV каждой группы

        Входные данные:
        ---------------
            stats : pandas.DataFrame
                    Данные полученных групп предиктора. Кол-во строк совпадает с кол-вом
                    уникальных значений предиктора.
                    Должен содержать столбцы: 'sample_count', 'target_count', 'groups'
        Возвращает:
        ---------------
            stats : pandas.DataFrame
                    Агрегированные данные по каждой группе

        """
        nothing = 10 ** -6
        stats = stats.groupby(['predictor', 'groups'],sort=False).agg({'type':'first',
                                                 'sample_count':'sum',
                                                 'target_count':'sum',
                                                 'value':['min', 'max']},
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


    # Служебные функции
    def _reset_state(self):
        self.trend_coefs = {}
        self.borders = {}
        self.predictors = []
        self.grouped = _GroupedPredictor()
        self.stats = _GroupedPredictor()

    def _get_nums_mask(self, x):
        if x.apply(lambda x:isinstance(x, str)).sum() == len(x):
            return pd.Series(False, index=x.index)
        else:
            mask = pd.to_numeric(x, errors='coerce').notna()
        return mask



class WoeTransformerRegularized(WoeTransformer):
    def __init__(self, min_sample_rate=0.05, min_count=3, alphas=[0], n_seeds=100):
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
        self.alphas = alphas
        self.alpha_values = {}
        self.n_seeds = n_seeds

    def fit(self, X, y, cat_values={}, alpha_values={}):
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
        self.regularization_stats = _GroupedPredictor()

        for col in X.columns:
            temp_alpha = self._cat_features_alpha_logloss(X[col].astype(str), y, self.alphas, self.n_seeds)
            self.alpha_values.update({col:temp_alpha})

        self._grouping(X, y)
        # Расчет WOE и IV
        self._fit_numeric(X, y)
        # Поиск потенциальных групп
        # Поиск "плохих" групп
        self._get_bad_groups()

        return self


    def _cat_features_alpha_logloss(self, x, y, alphas, seed=100):
        """
        функция расчета IV, GINI и logloss для категориальных переменных с корректировкой целевой по alpha

        """
        # задаем промежуточную функцию для WOE преобразования переменной из исходного датафрейма
                # по рассчитанным WOE из IVWOE
        def calc_woe_i(row_value, stats):
            return stats.loc[stats['groups']==row_value, 'WOE'].values[0]

        predictor = x.name
        target = y.name
        df = pd.DataFrame({predictor: x.values,
                           target: y.values})
        df[predictor] = df[predictor].fillna('NO_INFO')
        L_logloss_mean = []
        GINI_IV_mean = []
        for alpha_i in alphas:
            logloss_i = []
            GINI_i = []
            IV_i = []
            for seed_i in range(seed):
                X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(x, y,
                                                        test_size=0.3, random_state=seed_i, stratify=y)
                # Группировка значений предиктора с текущим alpha
                df_i = self._group_single(X_train, y_train, alpha_i)
                df_i['groups'] = df_i['value'].fillna('пусто')
                df_i['type'] = 'cat'
                # Обучение и применение группировки к обучающему набору
                WOE_i = self._fit_single(X_train, y_train, df_i)
                X_test_WOE = self._transform_single(X_test, WOE_i)

                roc_auc_i = sk.metrics.roc_auc_score(y_test, X_test_WOE)
                # Подстановка регуляризованной доли целевой вместо каждой группы
                target_transformed = X_test.replace(dict(zip(df_i['groups'], df_i['target_rate'])))
                # Запись значений
                logloss_i.append(sk.metrics.log_loss(y_test, target_transformed.fillna(0)))
                IV_i.append(WOE_i['IV'].sum())
                GINI_i.append(abs(2 * roc_auc_i - 1))
            # Запись средних значений
            L_logloss_mean.append([alpha_i, np.mean(logloss_i)])
            GINI_IV_mean.append([alpha_i, np.mean(GINI_i), np.mean(IV_i)])

        alpha_GINI_IV = pd.DataFrame(GINI_IV_mean, columns=['alpha', 'GINI', 'IV'])
        alpha_GINI_IV.insert(0, 'predictor', predictor)
        self.regularization_stats = self.regularization_stats.append(alpha_GINI_IV)

        # Индекс значения alpha с наименьшим логлоссом
        min_logloss_ind = np.argmin(L_logloss_mean, axis=0)[1]
        alpha_opt = L_logloss_mean[min_logloss_ind][0]

        return alpha_opt

    # def _plot_regularize_single(self, predictor):
    #     reg_stats = self.regularization_stats.get_predictor(predictor)
    #     fig = plt.figure(figsize=(16, 8))
    #     ax = fig.add_subplot(111)
    #     ax.plot(reg_stats['alpha'], reg_stats['logloss'], label = 'logloss_test', marker='o', ms = 3, color = 'red')
    #     ax2 = ax.twinx()
    #     ax2.plot(df_cat_features_alpha_GINI_IV['alpha'], df_cat_features_alpha_GINI_IV['IV'], label = 'IV_train', marker='o', ms = 3, color = 'blue')
    #     ax2.plot(df_cat_features_alpha_GINI_IV['alpha'], df_cat_features_alpha_GINI_IV['GINI'], label = 'GINI_test', marker='o', ms = 3, color = 'green')

    #     ax_y_step = (max(reg_stats['logloss']) - min(reg_stats['logloss'])) * 0.1
    #     ax_y_min = min(reg_stats['logloss']) - ax_y_step
    #     ax_y_max = max(reg_stats['logloss']) + ax_y_step
    #     ax.set_ylim(ax_y_min, ax_y_max)

    #     ax2_y_step = (max(max(df_cat_features_alpha_GINI_IV['IV']), max(df_cat_features_alpha_GINI_IV['GINI'])) - min(min(df_cat_features_alpha_GINI_IV['IV']), min(df_cat_features_alpha_GINI_IV['GINI']))) * 0.1
    #     ax2_y_min = min(min(df_cat_features_alpha_GINI_IV['IV']), min(df_cat_features_alpha_GINI_IV['GINI'])) - ax2_y_step
    #     ax2_y_max = max(max(df_cat_features_alpha_GINI_IV['IV']), max(df_cat_features_alpha_GINI_IV['GINI'])) + ax2_y_step
    #     ax2.set_ylim(ax2_y_min, ax2_y_max)

    #     ax.tick_params(axis="x", labelsize=12)
    #     ax2.tick_params(axis="x", labelsize=12)
    #     ax.set_xlabel('alpha', fontsize=16)
    #     ax.set_ylabel('logloss', fontsize=16)
    #     ax2.set_ylabel('GINI and IV', fontsize=16)
    #     ax.legend(loc = "upper left")
    #     ax2.legend(loc = "upper right")
    #     plt.grid(True)
    #     plt.title('Распределение logloss, GINI и IV от значения alpha', fontsize=20)
    #     plt.show()
