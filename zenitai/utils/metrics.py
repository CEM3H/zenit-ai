"""
Модуль с функциями для расчета метрик - Gini, PSI и т.п.

"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from typing import Any


def calc_PSI(exp, act):
    """
    Расчет значений PSI для сравнения распределений одной переменной в двух выборках
    Предполагается, что количественные переменные будут представлены в виде
    соответствующих групп (после равномерного биннинга или WOE-преобразования).
    Категориальные переменные могут передаваться без предварительных трансформаций

    Parameters
    ----------
    exp : pandas.Series
            Значения предиктора из первой выборки ("ожидаемые" в терминологии PSI)
    act : pandas.Series
            Значения предиктора из второй выборки ("наблюдаемые" в терминологии PSI)
    Returns
    --------
    df : pandas.DataFrame
            Таблица с ожидаемыми и наблюдаемыми частотами и рассчитанных PSI по каждой группе

    """
    # Расчет долей каждой категории в обеих выборках
    exp = exp.value_counts(normalize=True).sort_index()
    act = act.value_counts(normalize=True).sort_index()
    # Соединение в один датафрейм
    df = pd.concat([exp, act], axis=1).fillna(0).reset_index()
    df.columns = ["group", "expected", "actual"]
    # Расчет PSI по каждой группе
    df["PSI"] = (df["actual"] - df["expected"]) * np.log(
        (df["actual"] + 0.000001) / (df["expected"] + 0.000001)
    )

    return df


def auc_to_gini(auc):
    """Расчет коэффициента Gini по значению ROC-AUC"""

    return 2 * auc - 1


def plot_roc(
    facts: list, preds: list, labels: list = None, suptitle: str = None
) -> None:
    """Отрисовка произвольного количества ROC-кривых на одном графике
    Например, чтобы показать качество модели на трейне и тесте

    Parameters
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
        raise ValueError("Length of `facts` is not equal to lenght of `preds`")
    if labels is None:
        labels = [f"label_{i}" for i in range(len(preds))]
    elif len(labels) < len(facts):
        labels.extend([f"label_{i}" for i in range(len(facts) - len(labels))])

    roc_list = []  # [(FPR_train, TPR_train, thresholds_train), ...]
    gini_list = []  # [Gini_train, Gini_validate, Gini_test]

    lw = 2  # толщина линий

    # Построение графика ROC
    plt.figure(figsize=(8, 8))  # размер рисунка
    for fact, p, label in zip(facts, preds, labels):
        fpr, tpr, _ = roc_curve(fact, p)
        gini = auc_to_gini(roc_auc_score(fact, p))
        roc_list.append((fpr, tpr))
        gini_list.append(gini)
        plt.plot(fpr, tpr, lw=lw, label=f"{label} (Gini = {gini:.2%})", alpha=0.5)

    plt.plot([0, 1], [0, 1], color="k", lw=lw, linestyle="--", alpha=0.5)

    plt.xlim([-0.05, 1.05])  # min и max значения по осям
    plt.ylim([-0.05, 1.05])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC curves", fontsize=16)
    plt.legend(loc="lower right", fontsize=16)
    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=20)
    plt.show()


def get_roc_curves(
    facts: list,
    preds: list,
    labels: list = None,
    suptitle: str = None,
    ax: plt.Axes = None,
    **kwargs: dict,
) -> plt.axes:
    """Отрисовка произвольного количества ROC-кривых на одном графике.
    Например, чтобы показать качество модели на трейне и тесте

    Parameters
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
        raise ValueError("Length of `facts` is not equal to lenght of `preds`")
    if labels is None:
        labels = [f"label_{i}" for i in range(len(preds))]
    elif len(labels) < len(facts):
        labels.extend([f"label_{i}" for i in range(len(facts) - len(labels))])

    roc_list = []  # [(FPR_train, TPR_train, thresholds_train), ...]
    gini_list = []  # [Gini_train, Gini_validate, Gini_test]

    # Задаем параметры графиков
    ax = ax or plt.gca()  # используем набор осей из входных данных или текущую фигуру
    lw = kwargs.pop("lw", 2)  # толщина линий
    alpha = kwargs.pop("alpha", 0.5)  # прозрачность

    # Построение графика ROC
    # fig, ax = plt.subplots(1,1, figsize=(8, 8), ) # размер рисунка

    for fact, p, label in zip(facts, preds, labels):
        fpr, tpr, _ = roc_curve(fact, p)
        gini = auc_to_gini(roc_auc_score(fact, p))
        roc_list.append((fpr, tpr))
        gini_list.append(gini)
        ax.plot(
            fpr, tpr, label=f"{label} (Gini = {gini:.2%})", lw=lw, alpha=alpha, **kwargs
        )

    ax.plot([0, 1], [0, 1], color="k", lw=lw, linestyle="--", alpha=alpha)

    ax.set_xlim([-0.05, 1.05])  # min и max значения по осям
    ax.set_ylim([-0.05, 1.05])
    ax.tick_params(labelsize=16)
    # ax.set_yticks(fontsize=16)
    ax.grid(True)
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title("ROC curves", fontsize=16)
    ax.legend(loc="lower right", fontsize=16)
    if suptitle is not None:
        ax.suptitle(suptitle, fontsize=20)
    return ax


def get_gini_and_auc(
    facts: list, preds: list, plot: bool = True, **kwargs: dict
) -> list:
    gini_list: list = []
    for f, p in zip(facts, preds):
        gini_list.append(auc_to_gini(roc_auc_score(f, p)))
    if plot:
        plot_roc(facts, preds)
    return gini_list


def calc_gini_lr(X, y):
    scores = []
    for i in tqdm(X.columns, desc="1-factor Gini"):
        X1, X2, y1, y2 = train_test_split(X[[i]], y, stratify=y, random_state=42)
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X1, y1)

        preds1 = lr.predict_proba(X1)[:, 1]
        preds2 = lr.predict_proba(X2)[:, 1]

        score1 = auc_to_gini(roc_auc_score(y1, preds1))
        score2 = auc_to_gini(roc_auc_score(y2, preds2))

        scores.append((score1, score2))

    res = pd.DataFrame.from_records(scores, columns=["gini_train", "gini_test"])
    res.insert(0, "predictor", X.columns)
    return res


@dataclass
class ModelMetrics:
    fitted_model: Any
    train_x: pd.DataFrame = field(repr=False)
    test_x: pd.DataFrame = field(repr=False)
    train_y: pd.Series = field(repr=False)
    test_y: pd.Series = field(repr=False)

    def __post_init__(self):
        # get model class
        self.model_class = str(self.fitted_model.__class__)[8:-2]
        # get predictions
        try:
            self.train_preds = self.fitted_model.predict_proba(self.train_x)[:, 1]
            self.test_preds = self.fitted_model.predict_proba(self.test_x)[:, 1]
        except AttributeError:
            self.train_x = sm.tools.add_constant(self.train_x)
            self.test_x = sm.tools.add_constant(self.test_x)
            self.train_preds = self.fitted_model.predict(self.train_x)
            self.test_preds = self.fitted_model.predict(self.test_x)
        # get predictor names
        self.predictors = list(self.train_x.columns)

        # get model coefficients
        # TODO: add options for non-regression models
        try:
            intercept = list(self.fitted_model.intercept_)
            coef = list(self.fitted_model.coef_[0])
            self.coefficients = intercept + coef
        except AttributeError:
            if hasattr(self.fitted_model, "params"):
                self.coefficients = self.fitted_model.params.values
            else:
                self.coefficients = []
        self.coefficients = list(self.coefficients)

        self.train_data_control_sum = self.train_x.sum().sum()
        self.test_data_control_sum = self.test_x.sum().sum()

    def get_metrics(self):
        metrics = self.__dict__.copy()
        to_del = [
            "fitted_model",
            "train_x",
            "train_y",
            "test_x",
            "test_y",
            "train_preds",
            "test_preds",
            "train_preds_bin",
            "test_preds_bin",
            "confusion_matrix",
            "TN",
            "FP",
            "FN",
            "TP",
        ]
        for i in to_del:
            try:
                metrics.pop(i)
            except KeyError:
                continue

        return metrics

    def dump_metrics(self, path):
        metrics_to_dump = dict(self.get_metrics())
        with open(path, "w") as file:
            json.dump(metrics_to_dump, file, indent=4, ensure_ascii=True)


@dataclass
class ModelMetricsClassification(ModelMetrics):

    binary_threshold: float = field(default=0.5)

    def __post_init__(self):
        super().__post_init__()

        self.train_preds_bin = (self.train_preds > self.binary_threshold).astype(int)
        self.test_preds_bin = (self.test_preds > self.binary_threshold).astype(int)

        # get classification_metrics
        # get confusion matrix
        self.confusion_matrix = pd.DataFrame(
            sklearn.metrics.confusion_matrix(self.test_y, self.test_preds_bin),
            index=["actual negative", "actual positive"],
            columns=["predicted negative", "predicted positive"],
        )
        # get Gini for train and test
        self.gini_train = (
            2 * sklearn.metrics.roc_auc_score(self.train_y, self.train_preds) - 1
        )
        self.gini_test = (
            2 * sklearn.metrics.roc_auc_score(self.test_y, self.test_preds) - 1
        )

        # get TN, FP, FN and TP
        self.TN, self.FP, self.FN, self.TP = self.confusion_matrix.values.ravel()

        # get other metrcis
        self.precision = self.TP / (self.TP + self.FP)
        self.sensitivity = self.TP / (self.TP + self.FN)
        self.specificity = self.TN / (self.TN + self.FP)

        self.samples = {
            "train": {
                "n_samples": float(self.train_x.shape[0]),
                "n_targets": float(self.train_y.sum()),
                "target_rate": self.train_y.sum() / len(self.train_y),
            },
            "test": {
                "n_samples": float(self.test_x.shape[0]),
                "n_targets": float(self.test_y.sum()),
                "target_rate": self.test_y.sum() / len(self.test_y),
            },
        }
        self.confusion = {
            "TN": float(self.TN),
            "FP": float(self.FP),
            "FN": float(self.FN),
            "TP": float(self.TP),
        }
