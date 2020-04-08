# Импорт библиотек
import numpy as np
import pandas as pd
pd.options.display.max_columns = 200
pd.options.display.max_rows = 500
pd.options.display.float_format = '{:,.4f}'.format

import math
from matplotlib import pyplot as plt
from IPython.display import display

def grouping(DF_data_i):
    """
    Агрегация данных по значениям предиктора
    DF_data_i[['predictor', 'target']] - таблица данных
    """
    DF_i = DF_data_i[['predictor', 'target']].groupby('predictor', as_index=False).count()
    DF_j = DF_data_i[['predictor', 'target']].groupby('predictor', as_index=False).sum()
    DF_grouping = DF_i.merge(DF_j, how='left', on='predictor')
    DF_grouping.columns = ['predictor', 'sample_count', 'target_count']
    DF_grouping['sample_rate'] = DF_grouping['sample_count'] / DF_grouping['sample_count'].sum()
    DF_grouping['target_rate'] = DF_grouping['target_count'] / DF_grouping['sample_count']
    
    return DF_grouping

def monotonic_borders(DF_grouping, p, min_sample_rate=0.05, min_count=3):
    """
    Определение оптимальных границ групп (монотонный тренд)
    DF_grouping - агрегированные данные по значениям предиктора
    DF_grouping[['predictor', 'sample_count', 'target_count', 'sample_rate', 'target_rate]]
    min_sample_rate - минимальный размер группы (доля от размера выборки)
    min_count - минимальное количество наблюдений каждого класса в группе
    """
    k01, k11 = (1, 1) if p[0] > 0 else (0, -1)
    L_borders = []
    min_ind = 0 # минимальный индекс. Начальные условия
    
    while min_ind < DF_grouping.shape[0]: # цикл по новым группам
        pd_gr_i = k01 # средняя pd в группе. Начальные условия (зависит от общего тренда)
        
        for j in range(min_ind, max(DF_grouping.index) + 1): # цикл по конечной границе
            DF_j = DF_grouping.loc[min_ind : j]
            sample_rate_i = DF_j['sample_rate'].sum() # доля выборки
            sample_count_i = DF_j['sample_count'].sum() # количество наблюдений
            target_count_i = DF_j['target_count'].sum() # количество целевых
            non_target_count_i = sample_count_i - target_count_i # количество нецелевых
            target_rate_i = target_count_i / sample_count_i
            
            if (sample_rate_i < min_sample_rate) or (target_count_i < min_count) or (non_target_count_i < min_count):
                continue # если граница не удовлетворяет условиям

            if target_rate_i * k11 < pd_gr_i * k11: # проверка оптимальности границы
                min_ind_i = j + 1
                pd_gr_i = target_rate_i
                score_j = DF_grouping.loc[j, 'predictor']
        
        min_ind = min_ind_i
        if len(L_borders) > 0 and score_j == L_borders[-1]: # Выход из цикла, если нет оптимальных границ
            break
        L_borders.append(score_j)

    # Проверка последней добавленной группы
    
    DF_j = DF_grouping.loc[DF_grouping['predictor'] > L_borders[-1]]
    sample_rate_i = DF_j['sample_rate'].sum() # доля выборки
    sample_count_i = DF_j['sample_count'].sum() # количество наблюдений
    target_count_i = DF_j['target_count'].sum() # количество целевых
    non_target_count_i = sample_count_i - target_count_i # количество нецелевых
    
    if (sample_rate_i < min_sample_rate) or (target_count_i < min_count) or (non_target_count_i < min_count):
        L_borders.remove(L_borders[-1]) # удаление последней границы
    
    return L_borders


def statistic(DF_groups):
    """
    Расчет статистики по группам
    DF_groups[['sample_count', 'target_count', 'groups']] - таблица данных по группам
    """
    nothing = 10 ** -6
    DF_statistic = DF_groups[['sample_count', 'target_count', 'groups']].groupby('groups', as_index=False, sort=False).sum()
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
   
    DF_statistic['WOE'] = [math.log(DF_statistic['non_event_rate_i'][i] 
                                    / (DF_statistic['event_rate_i'][i] + nothing) + nothing) 
                           for i in DF_statistic.index]
    DF_statistic['IV'] = DF_statistic['WOE'] * (DF_statistic['non_event_rate_i'] - DF_statistic['event_rate_i'])
   
    DF_statistic = DF_statistic.merge(DF_groups[['type', 'groups']].drop_duplicates(), how='left', on='groups')
   
    return DF_statistic


def group_plot(DF_result, L_cols=['sample_rate', 'target_rate', 'WOE']):
    """
    Построение графика по группировке предиктора
    DF_result - таблица данных
    L_cols - список названий столбцов
    L_cols = ['sample_rate', 'target_rate', 'WOE']
    """
    [sample_rate, target_rate, WOE] = L_cols
    
    fig, ax_pd = plt.subplots(figsize=(8, 5))
    
    x2 = [DF_result[sample_rate][:i].sum() for i in range(DF_result.shape[0])] + [1] # доля выборки с накоплением
    x = [np.mean(x2[i:i + 2]) for i in range(len(x2) - 1)] # средняя точка в группах
    woe = list(DF_result[WOE])
    height = list(DF_result[target_rate]) # проблемность в группе
    width = list(DF_result[sample_rate]) # доля выборки на группу

    bar_pd = ax_pd.bar(x=x, height=height, width=width, color=[0, 122/255, 123/255], label='Группировка', alpha=0.7)

    ax_woe = ax_pd.twinx()
    line_woe = ax_woe.plot(x, woe, lw=2, color=[37/255, 40/255, 43/255], label='woe', marker='o')
    line_0 = ax_woe.plot([0, 1], [0, 0], lw=1, color=[37/255, 40/255, 43/255], linestyle='--')

    plt.xlim([0, 1])
    plt.xticks(x2, [round(i, 2) for i in x2], fontsize=12)
    ax_pd.grid(True)
    ax_pd.set_xlabel('Доля выборки', fontsize=16)
    ax_pd.set_ylabel('pd', fontsize=16)
    ax_woe.set_ylabel('woe', fontsize=16)

    # расчет границ графика и шага сетки
    max_woe = max([int(abs(i)) + 1 for i in woe])
    
    max_pd = max([int(i * 10) + 1 for i in height]) / 10
    
    ax_pd.set_ylim([0, max_pd])
    ax_woe.set_ylim([-max_woe, max_woe])

    ax_pd.set_yticks([round(i, 2) for i in np.linspace(0, max_pd, 11)])
    ax_woe.set_yticks([round(i, 2) for i in np.linspace(-max_woe, max_woe, 11)])

    plt.title('Группировка предиктора', fontsize=18)

    ax_pd.legend(loc=[0.2, -0.25], fontsize=14)
    ax_woe.legend(loc=[0.6, -0.25], fontsize=14)
    
    # для категориальных
    n_cat = DF_result.loc[DF_result['type'] == 'cat'].shape[0]

    if n_cat > 0:
        bar_pd = ax_pd.bar(x=x[-n_cat:], height=height[-n_cat:], width=width[-n_cat:], color='m', label='Категориальные')
        ax_pd.legend(loc=[0.15, -0.33], fontsize=14)

    plt.show()


# %% ExecuteTime={"start_time": "2020-03-25T11:06:21.844897Z", "end_time": "2020-03-25T11:06:21.855955Z"}
def woeTransformer(x, y, cat_values=[], min_sample_rate=0.05, min_count=3, monotonic=True, plot=True):
    """
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
    DF_data_i['predictor'] = x
    DF_data_i['target'] = y
    
    # Агрегация данных по значениям предиктора
    DF_data_gr = grouping(DF_data_i)
    
    # Проверка категориальных групп (возможные дополнительные категории)
    # 1) возможные дополнительные категории
    DF_i1 = DF_data_gr.loc[DF_data_gr['sample_rate'] > min_sample_rate].loc[~DF_data_gr['predictor'].isin(cat_values)]
    DF_i2 = DF_data_gr.loc[~DF_data_gr['predictor'].isin(cat_values)]
    L = []
    for i in DF_i2['predictor']:
        try:
            L.append(np.inf < i)
        except:
            L.append(True)
    DF_i2 = DF_i2.loc[pd.Series(L, index=DF_i2.index)]
    DF_i = DF_i1.append(DF_i2, ignore_index=True).drop_duplicates()
    if DF_i.shape[0] > 0:
        print('Возможно эти значения предиктора тоже являются категориальными:')
        display(DF_i)
    
    try:
        # Выделение числовых значений предиктора
        DF_data_gr_2 = DF_data_gr.loc[~DF_data_gr['predictor'].isin(cat_values)].reset_index(drop=True)
        if DF_data_gr_2.shape[0] > 0:
            DF_data_gr_2['predictor'] = DF_data_gr_2['predictor'].astype('float')

            # Определение тренда по числовым значениям
            DF_i = DF_data_i.loc[~DF_data_i['predictor'].isin(cat_values)]
            p = np.polyfit(DF_i['predictor'].astype('float'), DF_i['target'], deg=1)

            # Определение оптимальных границ групп
            L_borders = monotonic_borders(DF_data_gr_2, p, min_sample_rate, min_count)

            # Применение границ
            DF_data_gr_2['groups'] = pd.cut(DF_data_gr_2['predictor'], [-np.inf] + L_borders + [np.inf])
            DF_data_gr_2['type'] = 'num'

        # Добавление данных по категориальным значениям
        DF_data_gr_2k = DF_data_gr.loc[DF_data_gr['predictor'].isin(cat_values)].reset_index(drop=True)
        DF_data_gr_2k['groups'] = DF_data_gr_2k['predictor'].copy()
        DF_data_gr_2k['type'] = 'cat'

        # Расчет статистики, WoE и IV по группам числовых значений
        if DF_data_gr_2.shape[0] > 0:
            DF_result = statistic(DF_data_gr_2.append(DF_data_gr_2k, ignore_index=True))
        else:
            DF_result = statistic(DF_data_gr_2k)

        # Проверка категориальных групп (категории, которые не удовлетворяют заданным ограничениям)
        DF_j = DF_result[(DF_result['sample_rate'] < min_sample_rate) 
                         | (DF_result['target_count'] < min_count) 
                         | (DF_result['sample_count'] - DF_result['target_count'] < min_count)]
        if DF_j.shape[0] > 0:
            print('Эти группы не удовлетворяют заданным ограничениям:')
            display(DF_j)

        # Построение графика
        if plot:
            group_plot(DF_result, ['sample_rate', 'target_rate', 'WOE'])

        return DF_result
    except:
        print('Ошибка при выполнении группировки')
		
		
		
		
		
def woe_apply(S_data, DF_groups):
    """
    Применение группировки и WoE-преобразования
    
    S_data - значения предиктора (pd.Series)
    DF_groups - данные о группировке
    X_woe - WoE-преобразования значений предиктора
    
    WoE = 0, если группа не встречалась в обучающей выборке
    """
    X_woe = S_data.copy()

    # predict по числовым значениям
    DF_num = DF_groups.loc[DF_groups['type'] == 'num']
    if DF_num.shape[0] > 0:
        for i in DF_num.index: # цикл по группам
            group_i = DF_num['groups'][i]
            woe_i = DF_num['WOE'][i]
            values = [woe_i if (type(S_data[j]) != str and S_data[j] in group_i) else X_woe[j] for j in S_data.index]
            X_woe = pd.Series(values, S_data.index, name='woe')


    # predict по категориальным значениям (может обновлять значения по числовым)
    DF_cat = DF_groups.loc[DF_groups['type'] == 'cat']
    if DF_cat.shape[0] > 0:
        for i in DF_cat.index: # цикл по группам
            group_i = DF_cat['groups'][i]
            woe_i = DF_cat['WOE'][i]
            values = []
            for j in S_data.index: # цикл по строкам
                try:
                    if S_data[j] == group_i:
                        values.append(woe_i)
                    else:
                        values.append(X_woe[j])
                except:
                    values.append(X_woe[j])
            X_woe = pd.Series(values, S_data.index, name='woe')
        
    # WoE = 0, если группа не встречалась в обучающей выборке
    X_woe.loc[~X_woe.isin(DF_groups['WOE'])] = 0.0
    
    return X_woe
