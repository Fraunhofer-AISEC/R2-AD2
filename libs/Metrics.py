import numpy as np

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_roc(pred_scores: np.ndarray, test_alarm: tuple) -> list:
    """
    Evaluate performance on metrics without a threshold
    :param pred_scores: network to be evaluated - or scores of already predicted data
    :param test_alarm: test data for network
    :return: [Area under Curve, Average Precision]
    """

    # We chose the AUC & AP
    all_results = [
        roc_auc_score(y_true=test_alarm[1], y_score=pred_scores),
        average_precision_score(y_true=test_alarm[1], y_score=pred_scores),
    ]

    return all_results


def roc_to_pandas(fpr: np.ndarray, tpr: np.ndarray, suffix: str, decimals: int = 3) -> pd.DataFrame:
    """
    Round the ROC results to save some computation time in TikZ
    :param fpr: false positive rate
    :param tpr: true positive rate
    :param suffix: string appended to the column names
    :param decimals: decimals kept
    :return: DataFrame with the rounded TPR&FPR values
    """

    out_df = pd.concat([
        pd.Series(fpr, name=f"fpr_{suffix}"),
        pd.Series(tpr, name=f"tpr_{suffix}")
    ], axis=1)

    # Round and delete duplicates (look for duplicates in the FPR)
    out_df = out_df.round(decimals=decimals)
    out_df = out_df.drop_duplicates(subset=f"fpr_{suffix}", ignore_index=True)

    return out_df
