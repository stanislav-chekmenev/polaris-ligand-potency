from collections import defaultdict
from typing import Tuple

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_error, r2_score


def mask_nan(y_true, y_pred):
    mask = ~np.isnan(y_true)
    y_true = np.array(y_true)[mask]
    y_pred = np.array(y_pred)[mask]
    return y_true, y_pred


def eval_potency(preds: dict[str, list], refs: dict[str, list]) -> Tuple[dict[str, float]]:
    """
    Eval pIC50 potency performance with MAE (already log10 transformed)

    Parameters
    ----------
    preds : dict[str, list]
        Dictionary of predicted pIC50 values for SARS-CoV-2 and MERS-CoV Mpro.
    refs : dict[str, list]
        Dictionary of reference pIC50 values for SARS-CoV-2 and MERS-CoV Mpro.

    Returns
    -------
    dict[str, float]
        Returns a dictonary of summary statistics
    """

    keys = {"pIC50 (SARS-CoV-2 Mpro)", "pIC50 (MERS-CoV Mpro)"}
    collect = defaultdict(dict)

    for k in keys:
        if k not in preds.keys() or k not in refs.keys():
            raise ValueError("required key not present")

        ref, pred = mask_nan(refs[k], preds[k])

        mae = mean_absolute_error(ref, pred)
        ktau = kendalltau(ref, pred)
        r2 = r2_score(ref, pred)

        # subchallenge statistics
        collect[k]["mean_absolute_error"] = mae
        collect[k]["kendall_tau"] = ktau.statistic
        collect[k]["r2"] = r2

    # compute macro average MAE
    macro_mae = np.mean([collect[k]["mean_absolute_error"] for k in keys])
    collect["aggregated"]["macro_mean_absolute_error"] = macro_mae

    # compute macro average R2
    macro_r2 = np.mean([collect[k]["r2"] for k in keys])
    collect["aggregated"]["macro_r2"] = macro_r2

    return collect
