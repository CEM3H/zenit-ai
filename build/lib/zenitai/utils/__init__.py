"""
Модуль с полезными функциями
"""

from . import functions
from .functions import (
    build_logistic_regression,
    check_feat_stats,
    cramers_corr,
    extract_sub_pmt_str,
    get_corr_matrices,
    get_predictions,
    get_worst_status,
    plot_hier_corr,
    positive_coef_drop,
    select_feats,
    select_feats_corr,
    select_features_corr,
    select_features_hierarchy,
    split_train_test_valid,
    styler_float,
)
from .metrics import (
    auc_to_gini,
    calc_gini_lr,
    calc_PSI,
    get_gini_and_auc,
    get_roc_curves,
    plot_roc,
)
from .tests import compare_results_test, compare_time_test
from .utils import (
    compare_series,
    generate_test_data,
    generate_train_data,
    read_from_mssql,
)

__all__ = [
    "get_corr_matrices",
    "select_feats",
    "cramers_corr",
    "positive_coef_drop",
    "get_predictions",
    "build_logistic_regression",
    "select_features_corr",
    "styler_float",
    "select_features_hierarchy",
    "check_feat_stats",
    "extract_sub_pmt_str",
    "select_feats_corr",
    "split_train_test_valid",
    "get_gini_and_auc",
    "get_worst_status",
    "plot_hier_corr",
    "calc_PSI",
    "auc_to_gini",
    "plot_roc",
    "get_roc_curves",
    "get_gini_and_auc",
    "calc_gini_lr",
    "read_from_mssql",
    "compare_series",
    "generate_train_data",
    "generate_test_data",
    "compare_results_test",
    "compare_time_test",
]
