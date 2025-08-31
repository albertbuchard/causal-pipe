"""Utilities for post-hoc residual covariance augmentation in SEM models."""

from typing import Optional, Dict, Any, List

import math
import re
import pandas as pd


def augment_residual_covariances_stepwise(
    data: pd.DataFrame,
    model_string: str,
    estimator: str = "MLR",
    std_lv: bool = True,
    max_add: int = 5,
    mi_cutoff: float = 10.0,
    sepc_cutoff: float = 0.10,
    delta_stop: float = 0.003,
    whitelist_pairs: Optional[pd.DataFrame] = None,
    forbid_pairs: Optional[pd.DataFrame] = None,
    same_occasion_regex: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Augment residual covariances for a fitted SEM model using a small stepwise search.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset used for fitting the model.
    model_string : str
        lavaan model specification.
    estimator : str, optional
        lavaan estimator, by default "MLR".
    std_lv : bool, optional
        Whether to standardize latent variables, by default True.
    max_add : int, optional
        Maximum number of covariances to add, by default 5.
    mi_cutoff : float, optional
        Minimum modification index threshold, by default 10.0.
    sepc_cutoff : float, optional
        Minimum |sepc.all| threshold, by default 0.10.
    delta_stop : float, optional
        Minimum improvement in CFI or RMSEA required to continue, by default 0.003.
    whitelist_pairs : Optional[pd.DataFrame], optional
        Optional whitelist of pairs with columns ``lhs`` and ``rhs``.
    forbid_pairs : Optional[pd.DataFrame], optional
        Optional blocklist of pairs with columns ``lhs`` and ``rhs``.
    same_occasion_regex : Optional[str], optional
        Regex to enforce same occasion pairs, by default None.
    verbose : bool, optional
        If True, prints progress information.

    Returns
    -------
    Dict[str, Any]
        A dictionary with the final model string, fit measures, history and added covariances.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    import rpy2.robjects.packages as rpackages

    pandas2ri.activate()
    if not rpackages.isinstalled("lavaan"):
        utils = rpackages.importr("utils")
        utils.install_packages("lavaan")
    ro.r("library(lavaan)")

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data)

    sem = ro.r["sem"]
    modindices = ro.r["modindices"]
    par_table = ro.r["parTable"]
    lav_inspect = ro.r["lavInspect"]
    param_est = ro.r["parameterEstimates"]

    def get_fit(fit) -> Dict[str, Optional[float]]:
        # lavInspect returns an R named vector. When numpy/pandas automatic
        # conversions are active (e.g. via pandas2ri.activate), this can be
        # converted into a plain ``numpy.ndarray`` which lacks the ``rx2``
        # method used below to access elements by name.  Using a local
        # converter with the default configuration ensures we keep an R
        # object that preserves the named structure.
        with localconverter(ro.default_converter):
            fm = lav_inspect(fit, "fit.measures")
        out: Dict[str, Optional[float]] = {}
        for key in [
            "cfi",
            "cfi.scaled",
            "tli",
            "tli.scaled",
            "rmsea",
            "rmsea.scaled",
            "srmr",
            "aic",
            "bic",
        ]:
            try:
                val = float(fm.rx2(key)[0])
                if math.isnan(val):
                    out[key] = None
                else:
                    out[key] = val
            except LookupError:
                out[key] = None
        return out

    def robust_mi_col(df: pd.DataFrame) -> str:
        if "mi.robust" in df.columns:
            return "mi.robust"
        if "mi.scaled" in df.columns:
            return "mi.scaled"
        return "mi"

    def same_occasion_ok(lhs: str, rhs: str, rx: Optional[str]) -> bool:
        if not rx:
            return True
        ml = re.search(rx, lhs)
        mr = re.search(rx, rhs)
        if not ml or not mr or ml.lastindex is None or mr.lastindex is None:
            return False
        return ml.group(ml.lastindex) == mr.group(mr.lastindex)

    def accept_pair(lhs: str, rhs: str) -> bool:
        if forbid_pairs is not None and not forbid_pairs.empty:
            fb = forbid_pairs
            if (
                ((fb["lhs"] == lhs) & (fb["rhs"] == rhs))
                | ((fb["lhs"] == rhs) & (fb["rhs"] == lhs))
            ).any():
                return False
        if whitelist_pairs is not None and not whitelist_pairs.empty:
            wl = whitelist_pairs
            return (
                ((wl["lhs"] == lhs) & (wl["rhs"] == rhs))
                | ((wl["lhs"] == rhs) & (wl["rhs"] == lhs))
            ).any()
        return True

    model_cur = model_string
    fit = sem(model_cur, data=r_data, estimator=estimator, **{"std.lv": std_lv})
    fit_init = get_fit(fit)
    fit_hist: List[Dict[str, Optional[float]]] = [fit_init]
    added: List[Dict[str, Any]] = []

    for step in range(1, max_add + 1):
        with localconverter(ro.default_converter + pandas2ri.converter):
            mi = ro.conversion.rpy2py(modindices(fit))
            pt = ro.conversion.rpy2py(par_table(fit))

        mi = mi[(mi["op"] == "~~") & (mi["lhs"] != mi["rhs"])]

        have = pt[(pt["op"] == "~~") & (pt["free"] > 0)][["lhs", "rhs"]]
        if not have.empty and not mi.empty:
            def already_have(row):
                lhs, rhs = row["lhs"], row["rhs"]
                return (
                    ((have["lhs"] == lhs) & (have["rhs"] == rhs))
                    | ((have["lhs"] == rhs) & (have["rhs"] == lhs))
                ).any()

            mi = mi[~mi.apply(already_have, axis=1)]

        dir_df = pt[(pt["op"] == "~") & (pt["free"] > 0)][["lhs", "rhs"]]
        if not dir_df.empty and not mi.empty:
            def in_dir(row):
                lhs, rhs = row["lhs"], row["rhs"]
                return (
                    ((dir_df["lhs"] == lhs) & (dir_df["rhs"] == rhs))
                    | ((dir_df["lhs"] == rhs) & (dir_df["rhs"] == lhs))
                ).any()

            mi = mi[~mi.apply(in_dir, axis=1)]

        if not mi.empty:
            def extra_filters(row):
                lhs, rhs = row["lhs"], row["rhs"]
                if not accept_pair(lhs, rhs):
                    return False
                if not same_occasion_ok(lhs, rhs, same_occasion_regex):
                    return False
                return True

            mi = mi[mi.apply(extra_filters, axis=1)]

        if mi.empty:
            break

        mi_col = robust_mi_col(mi)
        mi = mi[
            mi[mi_col].notna()
            & (mi[mi_col] >= mi_cutoff)
            & mi["sepc.all"].notna()
            & (mi["sepc.all"].abs() >= sepc_cutoff)
        ]

        if mi.empty:
            break

        mi = mi.sort_values(by=[mi_col, "sepc.all"], ascending=[False, False])
        cand = mi.iloc[0]
        add_line = f"{cand.lhs} ~~ {cand.rhs}"
        model_try = model_cur + "\n" + add_line

        try:
            fit_new = sem(model_try, data=r_data, estimator=estimator, **{"std.lv": std_lv})
        except Exception:
            break

        fit_old = fit_init
        fit_new_m = get_fit(fit_new)

        cfi_old = (
            fit_old.get("cfi.scaled") if fit_old.get("cfi.scaled") is not None else fit_old.get("cfi")
        )
        cfi_new = (
            fit_new_m.get("cfi.scaled") if fit_new_m.get("cfi.scaled") is not None else fit_new_m.get("cfi")
        )
        rmsea_old = (
            fit_old.get("rmsea.scaled") if fit_old.get("rmsea.scaled") is not None else fit_old.get("rmsea")
        )
        rmsea_new = (
            fit_new_m.get("rmsea.scaled") if fit_new_m.get("rmsea.scaled") is not None else fit_new_m.get("rmsea")
        )
        d_cfi = (cfi_new or 0.0) - (cfi_old or 0.0)
        d_rmsea = (rmsea_old or 0.0) - (rmsea_new or 0.0)

        with localconverter(ro.default_converter + pandas2ri.converter):
            pe = ro.conversion.rpy2py(param_est(fit_new, standardized=True))
        heywood = (
            (pe["op"] == "~~")
            & (pe["lhs"] == pe["rhs"])
            & (pe["std.all"] < 0)
        ).any()

        bic_old = fit_old.get("bic")
        bic_new = fit_new_m.get("bic")
        bic_diff = None
        if bic_old is not None and bic_new is not None:
            bic_diff = bic_new - bic_old

        if heywood or (
            d_cfi < delta_stop
            and d_rmsea < delta_stop
            and (bic_diff is None or bic_diff >= -2)
        ):
            break

        model_cur = model_try
        fit = fit_new
        fit_init = fit_new_m
        added.append(
            {
                "lhs": str(cand.lhs),
                "rhs": str(cand.rhs),
                "mi": float(cand[mi_col]),
                "sepc.all": float(cand["sepc.all"]),
                "mi_col": mi_col,
                "step": len(added) + 1,
            }
        )
        fit_hist.append(fit_new_m)
        if verbose:
            print(f"Added {add_line}")

    final_model_string = model_cur

    return {
        "final_model_string": final_model_string,
        "fit_measures": fit_init,
        "initial_fit_measures": fit_hist[0],
        "added_covariances": added,
        "fit_history": fit_hist,
    }
