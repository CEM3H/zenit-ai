"""
Модуль с функциями для расчета метрик - Gini, PSI и т.п.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def calc_PSI(exp, act):
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


def auc_to_gini(auc):
    return 2 * auc - 1


def plot_roc(facts:list, preds:list, labels:list=None, suptitle:str=None):
    """ Отрисовка произвольного количества ROC-кривых на одном графике
    Например, чтобы показать качество модели на трейне и тесте
    Входные данные:
    ---------------
        facts : list of arrays
                Список, состоящий из массивов фактических меток классов (0 или 1)
        preds : list of arrays
                Список состоящий из массивов предсказаний классификатора (результаты метода
                `predict_proba()`)
        labels : list, default None
                Список меток для графиков
        suptitle : str, default None
                Над-заголовок для графика
    """
    if not len(facts) == len(preds):
        raise ValueError('Length of `facts` is not equal to lenght of `preds`')
    if labels is None:
        labels = [f'label_{i}' for i in range(len(preds))]
    elif len(labels) < len(facts):
        labels.extend([f"label_{i}" for i in range(len(facts) - len(labels))])

    roc_list = [] # [(FPR_train, TPR_train, thresholds_train), ...]
    gini_list = [] # [Gini_train, Gini_validate, Gini_test]

    lw=2 # толщина линий

    # Построение графика ROC
    plt.figure(figsize=(8, 8)) # размер рисунка
    for fact, p, label in zip(facts, preds, labels):
        fpr, tpr, _ = roc_curve(fact, p)
        gini = auc_to_gini(roc_auc_score(fact, p))
        roc_list.append((fpr, tpr))
        gini_list.append(gini)
        plt.plot(fpr, tpr, lw=lw,
                 label=f'{label} (Gini = {gini:.2%})', alpha=0.5)

    plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--', alpha=0.5)

    plt.xlim([-0.05, 1.05]) # min и max значения по осям
    plt.ylim([-0.05, 1.05])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC curves', fontsize=16)
    plt.legend(loc='lower right', fontsize=16)
    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=20)
    plt.show()


def get_roc_curves(facts:list, preds:list, labels:list=None, suptitle:str=None, ax=None, **kwargs):
    """ Отрисовка произвольного количества ROC-кривых на одном графике
    Например, чтобы показать качество модели на трейне и тесте
    Входные данные:
    ---------------
        facts : list of arrays
                Список, состоящий из массивов фактических меток классов (0 или 1)
        preds : list of arrays
                Список состоящий из массивов предсказаний классификатора (результаты метода
                `predict_proba()`)
        labels : list, default None
                Список меток для графиков
        suptitle : str, default None
                Над-заголовок для графика
        ax : matplotlib Axes object
        **kwargs : параметры для передачи в plt.plot

    """
    if not len(facts) == len(preds):
        raise ValueError('Length of `facts` is not equal to lenght of `preds`')
    if labels is None:
        labels = [f'label_{i}' for i in range(len(preds))]
    elif len(labels) < len(facts):
        labels.extend([f"label_{i}" for i in range(len(facts) - len(labels))])

    roc_list = [] # [(FPR_train, TPR_train, thresholds_train), ...]
    gini_list = [] # [Gini_train, Gini_validate, Gini_test]

    # Задаем параметры графиков
    ax = ax or plt.gca()   # используем набор осей из входных данных или текущую фигуру
    lw = kwargs.pop('lw', 2)  # толщина линий
    alpha = kwargs.pop('alpha', 0.5) # прозрачность

    # Построение графика ROC
    # fig, ax = plt.subplots(1,1, figsize=(8, 8), ) # размер рисунка

    for fact, p, label in zip(facts, preds, labels):
        fpr, tpr, _ = roc_curve(fact, p)
        gini = auc_to_gini(roc_auc_score(fact, p))
        roc_list.append((fpr, tpr))
        gini_list.append(gini)
        ax.plot(fpr, tpr, label=f'{label} (Gini = {gini:.2%})', lw=lw, alpha=alpha, **kwargs)

    ax.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--', alpha=alpha)

    ax.set_xlim([-0.05, 1.05]) # min и max значения по осям
    ax.set_ylim([-0.05, 1.05])
    ax.tick_params(labelsize=16)
    # ax.set_yticks(fontsize=16)
    ax.grid(True)
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC curves', fontsize=16)
    ax.legend(loc='lower right', fontsize=16)
    if suptitle is not None:
        ax.suptitle(suptitle, fontsize=20)
    return ax


def get_gini_and_auc(facts:list, preds:list, plot=True, **kwargs):
    gini_list = []
    for f, p in zip(facts, preds):
        gini_list.append(auc_to_gini(roc_auc_score(f, p)))
    if plot:
        plot_roc(facts, preds, **kwargs)
    return gini_list


def calc_gini_lr(X, y):
    scores = []
    for i in tqdm(X.columns, desc='1-factor Gini'):
        X1, X2, y1, y2 = train_test_split(X[[i]], y, stratify=y, random_state=42)
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X1, y1)

        preds1 = lr.predict_proba(X1)[:, 1]
        preds2 = lr.predict_proba(X2)[:, 1]

        score1 = auc_to_gini(roc_auc_score(y1, preds1))
        score2 = auc_to_gini(roc_auc_score(y2, preds2))

        scores.append((score1, score2))

    res = pd.DataFrame.from_records(scores, columns=['gini_train', 'gini_test'])
    res.insert(0, 'predictor', X.columns)
    return res
