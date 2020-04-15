#!/usr/bin/env python
# coding: utf-8

# # Функция группировки и WOE-преобразования

# ### проверка на примере данных по автокредитам

# In[ ]:
# ## Changelog

# ### v1
# * исключение категориальных значений (пустышки и несравнимые значения - для монотонного тренда!)
# **При группировке учитываются следующие параметры:**
# * минимальный размер группы (доля от размера выборки)
# * минимальное количество наблюдений каждого класса в группе
# * монотонный тренд
# ### v2
# * .loc в monotonic_borders
# * объединение def statistic и def woe_iv
# * подсчет статистики по категориальным значениям
# * категориальные значения отдельно
#     * на графике группировки линия 1% среднее по проблемности
# ### v3 - доработка

# * обработка кейсов:
#     * только категориальные переменные
#     * деление на 0 при расчете WoE (nothing = 10 ** -6)
# * monotonic_borders:
#     * цикл while по группам
#         * а если нет оптимальных границ? (такое возможно?)
#     * максимальная конечная граница
#     * исключено: удаление дубликатов и сортировка (теперь нет необходимости)
#     * проверка последней добавленной группы
# * woeTransformer:
#     * проверка дополнительных категорий type -> try / except
#     * np.polyfit (.astype('float') внутрь функции, чтобы не менять тип данных)
#     * try / except для группировки числовых
#     * генерация DF_result (не добавление к пустому DF_data_gr_2)
#     * Проверка соответствия всех групп заданным условиям
#     * IPython.display.display(pd.DataFrame) для красивого вывода таблиц
# * значение предиктора np.nan???
# * переобозначение внутренних имен для упрощения проверки

# ### v3.1 - быстродействие
# * Существенно расширены docstring-и и комментарии к функицям
# * monotonic_borders:
# 	  * переписан алгоритм расчета монотонных границ (цикл for -> скользящие функции pandas)
#     * L_borders переименовано в R_borders, т. к. функция возвращает правые границы групп
# * grouping
#     * добавлена опция округления при группировке (параметр low_acc)
#		Параметр регулирует количество знаков после запятой после округления
# * woeTransformer
# 	  * изменена логика расчета возможных новых категорий (циклы -> индексация pandas)
#     * неиспользуемый параметр monotonic_ заменен на low_accuracy, который определяет, 
#       будет  ли включено округление значений предиктора и до скольки символов
# * group_plot
#	  * убран параметр L_cols

# ### v3.2 - исправление ошибок и логики
# * woe_apply
#    * исправлена обработка  новых категорий при применении обученных категорий к новым даннмы (в  v3.1 новые категории выпадаи из обработки) 
# *  monotonic_borders
#	* добавлена поправка ня точность хранения float - теперь при сравнении sample_rate c минимально требуемым из последнего вычитается 10^-9
# * woeTransformer
#	* добавлен параметр verbose, который регулирует показ доп. информации при группировке (по умолчанию True)


# ## Библиотеки
# Импорт библиотек
import math
import time

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from IPython.display import display
from scipy.stats import sem
from IPython.display import display


# ## Функция улучшенная

# ### Группировка
def grouping(DF_data_i, low_acc=False):
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
    # Округение, если аргумент принимает допустимые значения
    if low_acc and type(low_acc) is int and low_acc > 0:
        DF_data_i = DF_data_i[['predictor', 'target']].round(low_acc)

    # Группировка и расчет показателей
    DF_grouping = DF_data_i.groupby('predictor')['target'].agg(['count', 'sum']).reset_index()
    DF_grouping.columns = ['predictor', 'sample_count', 'target_count']
    DF_grouping['sample_rate'] = DF_grouping['sample_count'] / DF_grouping['sample_count'].sum()
    DF_grouping['target_rate'] = DF_grouping['target_count'] / DF_grouping['sample_count']

    return DF_grouping


# ### Монотонные границы
def monotonic_borders(DF_grouping, p, min_sample_rate=0.05, min_count=3):
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
        DF_iter['check'] = ((DF_iter['sample_rate'] >= min_sample_rate - 10 ** -9)
                            & (DF_iter['target_count'] >= min_count)
                            & (DF_iter['non_target_count'] >= min_count))

        # Расчет базы для проверки оптимальности границы
        # В зависимости от тренда считается скользящий _вперед_ минимум или максимум 
        # (в расчете участвуют все наблюдения от текущего до последнего)
        if k11 == 1:
            DF_iter['pd_gr'] = DF_iter['target_rate'][::-1].rolling(len(DF_iter), min_periods=0).min()[::-1]
        else:
            DF_iter['pd_gr'] = DF_iter['target_rate'][::-1].rolling(len(DF_iter), min_periods=0).max()[::-1]

        # Проверка оптимальности границы
        DF_iter['opt'] = DF_iter['target_rate'] == DF_iter['pd_gr']
        DF_iter = pd.concat([DF_j[['predictor']], DF_iter], axis=1)
        try:
            min_ind = DF_iter.loc[(DF_iter['check'] == True)
                                  & (DF_iter['opt'] == True)
            , 'target_rate'].index.values[0]
            pd_gr_i = DF_iter.loc[min_ind, 'target_rate']
            score_j = DF_iter.loc[min_ind, 'predictor']
            if len(R_borders) > 0 and score_j == R_borders[-1]:  # Выход из цикла, если нет оптимальных границ
                break
        except:
            break
        min_ind += 1
        R_borders.append(score_j)

    # Проверка последней добавленной группы
    DF_iter = DF_grouping.loc[DF_grouping['predictor'] > R_borders[-1]]
    sample_rate_i = DF_iter['sample_rate'].sum()  # доля выборки
    sample_count_i = DF_iter['sample_count'].sum()  # количество наблюдений
    target_count_i = DF_iter['target_count'].sum()  # количество целевых
    non_target_count_i = sample_count_i - target_count_i  # количество нецелевых

    if (sample_rate_i < min_sample_rate) or (target_count_i < min_count) or (non_target_count_i < min_count):
        R_borders.remove(R_borders[-1])  # удаление последней границы
    return R_borders


### Статистика
def statistic(DF_groups):
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
        DF_statistic : pandas.DataFrame
                Агрегированные данные по каждой группе
            
    """
    nothing = 10 ** -6
    DF_statistic = DF_groups[['sample_count', 'target_count', 'groups']].groupby('groups', as_index=False,
                                                                                 sort=False).sum()
    DF_statistic_min = DF_groups[['predictor', 'groups']].groupby('groups', as_index=False, sort=False).min()
    DF_statistic_max = DF_groups[['predictor', 'groups']].groupby('groups', as_index=False, sort=False).max()
    DF_statistic['min'] = DF_statistic_min['predictor']
    DF_statistic['max'] = DF_statistic_max['predictor']
    DF_statistic['sample_rate'] = DF_statistic['sample_count'] / DF_statistic['sample_count'].sum()
    DF_statistic['target_rate'] = DF_statistic['target_count'] / DF_statistic['sample_count']

    # Расчет WoE и IV
    samples_num = DF_statistic['sample_count'].sum()
    events = DF_statistic['target_count'].sum()
    non_events = samples_num - events

    DF_statistic['non_events_i'] = DF_statistic['sample_count'] - DF_statistic['target_count']
    DF_statistic['event_rate_i'] = DF_statistic['target_count'] / (events + nothing)
    DF_statistic['non_event_rate_i'] = DF_statistic['non_events_i'] / (non_events + nothing)

    DF_statistic['WOE'] = np.log(DF_statistic['non_event_rate_i']
                                 / (DF_statistic['event_rate_i'] + nothing)
                                 + nothing)

    DF_statistic['IV'] = DF_statistic['WOE'] * (DF_statistic['non_event_rate_i'] - DF_statistic['event_rate_i'])

    DF_statistic = DF_statistic.merge(DF_groups[['type', 'groups']].drop_duplicates(), how='left', on='groups')

    return DF_statistic


### Графики
def group_plot(DF_result):
    """
    Построение графика по группировке предиктора
    
    Входные данные:
    ---------------
        DF_result : pandas.DataFrame
                Статистика по каждой группе (результат работы функции statistic):
                    минимальное, максимальное значение, доля от общего объема выборки,
                    количество и доля целевых и нецелевых событий в каждой группе,
                    WOE и IV каждой группы
                Должен содержать столбцы: 'sample_rate', 'target_rate', 'WOE'
                
    Возвращает:
    ---------------
        None
                Не возвращает ничего
        
    """
    ## Расчеты
    sample_rate, target_rate, WOE = ['sample_rate', 'target_rate', 'WOE']

    x2 = [DF_result[sample_rate][:i].sum()
          for i in range(DF_result.shape[0])] + [1]  # доля выборки с накоплением
    x = [np.mean(x2[i:i + 2]) for i in range(len(x2) - 1)]  # средняя точка в группах

    # Выделение нужной информации для компактности
    woe = list(DF_result[WOE])
    height = list(DF_result[target_rate])  # проблемность в группе
    width = list(DF_result[sample_rate])  # доля выборки на группу

    ## Визуализация
    fig, ax_pd = plt.subplots(figsize=(8, 5))

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
    ax_pd.legend(loc=[0.2, -0.25], fontsize=14)
    # Границы и сетка для графика WOE
    ax_woe.set_ylim([-max_woe, max_woe])
    ax_woe.set_yticks([round(i, 2) for i in np.linspace(-max_woe, max_woe, 11)])
    ax_woe.legend(loc=[0.6, -0.25], fontsize=14)

    plt.title('Группировка предиктора', fontsize=18)

    # Для категориальных
    n_cat = DF_result.loc[DF_result['type'] == 'cat'].shape[0]

    if n_cat > 0:
        bar_pd = ax_pd.bar(x=x[-n_cat:], height=height[-n_cat:], width=width[-n_cat:], color='m',
                           label='Категориальные')
        ax_pd.legend(loc=[0.15, -0.33], fontsize=14)

    plt.show()


# ### Трансформер

def woeTransformer(x, y,
                   cat_values=[],
                   min_sample_rate=0.05,
                   min_count=3,
                   low_accuracy=None,
                   plot=True,
                   verbose=True):
    """
    Группировка значений предиктора, определение оптимальных границ и расчет WOE и IV
    
    Входные данные:
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
        low_accuracy : int, default None
                Режим пониженной точности (округление при группировке)
                Если None, то предиктор не округляется.
                Если целое неотрицательное число, параметр используется для определения 
                количества знаков после запятой, остальные значения игнорируются
        plot : bool, default True
                Включение/выключение визуализации группировки
        verbose : bool, default True
		        Включение.выключение доп. информации по группировке
                
    Возвращает:
    ---------------
        DF_result : pandas.DataFrame
                Таблица с итоговой группировкой и статистикой
                
    """
    # Обработка входных данных
    DF_data_i = pd.DataFrame({'predictor':x,
                              'target':y})

    # Агрегация данных по значениям предиктора
    DF_data_gr = grouping(DF_data_i, low_accuracy)

    # Проверка категориальных групп (возможные дополнительные категории)
    if verbose:
        ## Выделение значений предиктора с достаточным кол-вом наблюдений и 
        ## не отмеченных, как категориальные
        DF_i1 = (DF_data_gr
            .loc[DF_data_gr['sample_rate'] > min_sample_rate]
            .loc[~DF_data_gr['predictor'].isin(cat_values)])

        ## Выделение всех значений предиктора, не отмеченных, как категориальные
        DF_i2 = DF_data_gr.loc[~DF_data_gr['predictor'].isin(cat_values)]

        ## Выбор значений: которые не равны бесконености и при этом не являются числами
        L = (~(DF_i2['predictor'] == np.inf)
             & (pd.to_numeric(DF_i2['predictor'], errors='coerce').isna())
             )

        DF_i2 = DF_i2.loc[L]
        # Объединение найденных значений в одну таблицу
        DF_i = DF_i1.append(DF_i2, ignore_index=True).drop_duplicates()
        if DF_i.shape[0] > 0:
            print('Возможно эти значения предиктора тоже являются категориальными:')
            display(DF_i)
    
    
    # Выделение числовых значений предиктора
    DF_data_gr_num = DF_data_gr.loc[~DF_data_gr['predictor'].isin(cat_values)].reset_index(drop=True)

    if DF_data_gr_num.shape[0] > 0:
        try:
            DF_data_gr_num['predictor'] = DF_data_gr_num['predictor'].astype('float')

            # Определение тренда по числовым значениям
            DF_i = DF_data_i.loc[~DF_data_i['predictor'].isin(cat_values)]
            p = np.polyfit(DF_i['predictor'].astype('float'), DF_i['target'], deg=1)
            # Определение оптимальных границ групп
            R_borders = monotonic_borders(DF_data_gr_num, p, min_sample_rate, min_count)
        except:
            print('Ошибка при расчете монотонных границ')
            
        try:
            # Применение границ
            DF_data_gr_num['groups'] = pd.cut(DF_data_gr_num['predictor'], [-np.inf] + R_borders + [np.inf])
            DF_data_gr_num['type'] = 'num'
        except:
            print('Ошибка при применении монотонных границ')

    # Добавление данных по категориальным значениям
    DF_data_gr_2k = DF_data_gr.loc[DF_data_gr['predictor'].isin(cat_values)].reset_index(drop=True)
    DF_data_gr_2k['groups'] = DF_data_gr_2k['predictor'].copy()
    DF_data_gr_2k['type'] = 'cat'

    try:
        # Расчет статистики, WoE и IV по группам числовых значений
        if DF_data_gr_num.shape[0] > 0:
            DF_result = statistic(DF_data_gr_num.append(DF_data_gr_2k, ignore_index=True))
        else:
            DF_result = statistic(DF_data_gr_2k)
    except:
        print('Ошибка при расчете статистики')
        

    # Проверка категориальных групп (категории, которые не удовлетворяют заданным ограничениям)
    if verbose:
        DF_j = DF_result.loc[(DF_result['sample_rate'] < min_sample_rate)
                                | (DF_result['target_count'] < min_count)
                                | (DF_result['sample_count']
                                - DF_result['target_count'] < min_count)]
        if DF_j.shape[0] > 0:
            print('Эти группы не удовлетворяют заданным ограничениям:')
            display(DF_j)
        # Построение графика
        if plot:
            group_plot(DF_result)

        return DF_result

		
def woe_apply(S_data, DF_groups):
    """
    Применение группировки и WoE-преобразования
    
    Входные данные:---------------
        S_data : pandas.Series
                Значения предиктора
        DF_groups : pandas.DataFrame
                Данные о группировке предиктора

    Возвращает:
    ---------------
        X_woe : pandas.DataFrame 
                WoE-преобразования значений предиктора
                WoE = 0, если группа не встречалась в обучающей выборке

    """
    X_woe = S_data.copy()
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
        X_woe_num = X_woe[X_woe.astype(str)
                              .str.replace('\.|\-', '')
                              .str.replace('e', '').str.isdecimal() &
                          (~X_woe.isin(cat_bounds))]
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
