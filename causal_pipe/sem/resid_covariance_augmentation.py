"""Utilities for post-hoc residual covariance augmentation in SEM models."""

from typing import Optional, Dict, Any, List

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

    ro.globalenv["data"] = r_data
    ro.globalenv["model_string"] = model_string
    ro.globalenv["estimator"] = estimator
    ro.globalenv["std_lv"] = std_lv
    ro.globalenv["mi_cutoff"] = mi_cutoff
    ro.globalenv["sepc_cutoff"] = sepc_cutoff
    ro.globalenv["delta_stop"] = delta_stop
    ro.globalenv["max_add"] = max_add

    if whitelist_pairs is not None and len(whitelist_pairs):
        wl = whitelist_pairs[["lhs", "rhs"]].copy()
        with localconverter(ro.default_converter + pandas2ri.converter):
            ro.globalenv["WL"] = ro.conversion.py2rpy(wl)
    else:
        ro.globalenv["WL"] = ro.r("NULL")
    if forbid_pairs is not None and len(forbid_pairs):
        fb = forbid_pairs[["lhs", "rhs"]].copy()
        with localconverter(ro.default_converter + pandas2ri.converter):
            ro.globalenv["FB"] = ro.conversion.py2rpy(fb)
    else:
        ro.globalenv["FB"] = ro.r("NULL")

    if isinstance(same_occasion_regex, str) and same_occasion_regex:
        ro.globalenv["same_regex"] = same_occasion_regex
    else:
        ro.globalenv["same_regex"] = ro.r("NA_character_")
    ro.globalenv["verbose"] = verbose

    ro.r(
        """
get_fit <- function(fit) {
  fm <- lavInspect(fit, "fit.measures")
  out <- list(
    cfi = unname(fm["cfi"]),
    cfi.scaled = if ("cfi.scaled" %in% names(fm)) unname(fm["cfi.scaled"]) else NA_real_,
    tli = unname(fm["tli"]),
    tli.scaled = if ("tli.scaled" %in% names(fm)) unname(fm["tli.scaled"]) else NA_real_,
    rmsea = unname(fm["rmsea"]),
    rmsea.scaled = if ("rmsea.scaled" %in% names(fm)) unname(fm["rmsea.scaled"]) else NA_real_,
    srmr = unname(fm["srmr"]),
    aic = unname(fm["aic"]),
    bic = unname(fm["bic"])
  )
  return(out)
}

robust_mi_col <- function(mi_df) {
  if ("mi.robust" %in% names(mi_df)) return("mi.robust")
  if ("mi.scaled" %in% names(mi_df)) return("mi.scaled")
  return("mi")
}

same_occasion_ok <- function(lhs, rhs, rx) {
  if (is.na(rx) || is.null(rx)) return(TRUE)
  ml <- regexec(rx, lhs); mr <- regexec(rx, rhs)
  sl <- regmatches(lhs, ml)[[1]]; sr <- regmatches(rhs, mr)[[1]]
  if (length(sl) < 3 || length(sr) < 3) return(FALSE)
  return(tail(sl,1) == tail(sr,1))
}

accept_pair <- function(lhs, rhs, WL, FB) {
  if (!is.null(FB)) {
    if (nrow(subset(FB, (lhs==.data$lhs & rhs==.data$rhs) | (lhs==.data$rhs & rhs==.data$lhs)))>0) return(FALSE)
  }
  if (!is.null(WL)) {
    return(nrow(subset(WL, (lhs==.data$lhs & rhs==.data$rhs) | (lhs==.data$rhs & rhs==.data$lhs)))>0)
  }
  return(TRUE)
}

model_cur <- model_string
fit <- sem(model_cur, data=data, std.lv=std_lv, estimator=estimator)
fit_init <- get_fit(fit)
fit_hist <- list(fit_init)
added <- list()

for (k in seq_len(max_add)) {
  mi <- modindices(fit)
  mi <- mi[mi$op=="~~" & mi$lhs != mi$rhs,, drop=FALSE]

  PT <- parTable(fit)
  have <- PT[PT$op=="~~" & PT$free>0, c("lhs","rhs")]
  if (nrow(have)>0L) {
    keep_idx <- rep(TRUE, nrow(mi))
    for (i in seq_len(nrow(mi))) {
      lhs <- mi$lhs[i]; rhs <- mi$rhs[i]
      if (nrow(subset(have, (lhs==.data$lhs & rhs==.data$rhs) | (lhs==.data$rhs & rhs==.data$lhs)))>0) keep_idx[i] <- FALSE
    }
    mi <- mi[keep_idx,,drop=FALSE]
  }

  dir <- PT[PT$op=="~" & PT$free>0, c("lhs","rhs")]
  if (nrow(dir)>0L && nrow(mi)>0L) {
    keep_idx <- rep(TRUE, nrow(mi))
    for (i in seq_len(nrow(mi))) {
      lhs <- mi$lhs[i]; rhs <- mi$rhs[i]
      if (nrow(subset(dir, (lhs==.data$lhs & rhs==.data$rhs) | (lhs==.data$rhs & rhs==.data$lhs)))>0) keep_idx[i] <- FALSE
    }
    mi <- mi[keep_idx,,drop=FALSE]
  }

  if (nrow(mi)>0L) {
    keep_idx <- rep(TRUE, nrow(mi))
    for (i in seq_len(nrow(mi))) {
      if (!accept_pair(mi$lhs[i], mi$rhs[i], WL, FB)) keep_idx[i] <- FALSE
      if (!same_occasion_ok(mi$lhs[i], mi$rhs[i], same_regex)) keep_idx[i] <- FALSE
    }
    mi <- mi[keep_idx,,drop=FALSE]
  }

  if (nrow(mi)==0L) break

  mi_col <- robust_mi_col(mi)
  mi <- mi[!is.na(mi[[mi_col]]) & mi[[mi_col]] >= mi_cutoff & !is.na(mi$sepc.all) & abs(mi$sepc.all) >= sepc_cutoff,, drop=FALSE]
  if (nrow(mi)==0L) break

  o <- order(-mi[[mi_col]], -abs(mi$sepc.all))
  cand <- mi[o[1],,, drop=FALSE]
  add_line <- paste0(cand$lhs, " ~~ ", cand$rhs)
  model_try <- paste(model_cur, add_line, sep="\n")

  fit_new <- try(suppressWarnings(sem(model_try, data=data, std.lv=std_lv, estimator=estimator)), silent=TRUE)
  if (inherits(fit_new, "try-error")) break

  fit_old <- fit_init
  fit_new_m <- get_fit(fit_new)

  cfi_old <- if (!is.na(fit_old$cfi.scaled)) fit_old$cfi.scaled else fit_old$cfi
  cfi_new <- if (!is.na(fit_new_m$cfi.scaled)) fit_new_m$cfi.scaled else fit_new_m$cfi
  rmsea_old <- if (!is.na(fit_old$rmsea.scaled)) fit_old$rmsea.scaled else fit_old$rmsea
  rmsea_new <- if (!is.na(fit_new_m$rmsea.scaled)) fit_new_m$rmsea.scaled else fit_new_m$rmsea

  d_cfi <- cfi_new - cfi_old
  d_rmsea <- rmsea_old - rmsea_new

  pe <- parameterEstimates(fit_new, standardized=TRUE)
  heywood <- any(pe$op=="~~" & pe$lhs==pe$rhs & pe$std.all < 0, na.rm=TRUE)

  if (heywood || (d_cfi < delta_stop && d_rmsea < delta_stop && (is.na(fit_old$bic) || is.na(fit_new_m$bic) || (fit_new_m$bic - fit_old$bic) >= -2))) {
    break
  } else {
    model_cur <- model_try
    fit <- fit_new
    fit_init <- fit_new_m
    added[[length(added)+1]] <- list(lhs=cand$lhs, rhs=cand$rhs, mi=cand[[mi_col]], sepc=cand$sepc.all, mi_col=mi_col, step=length(added)+1)
    fit_hist[[length(fit_hist)+1]] <- fit_new_m
    if (verbose) cat(paste0("Added ", add_line, "\n"))
  }
}

final_fit <- fit_init
"""
    )

    fit_measures = dict(
        zip(
            [
                "cfi",
                "cfi.scaled",
                "tli",
                "tli.scaled",
                "rmsea",
                "rmsea.scaled",
                "srmr",
                "aic",
                "bic",
            ],
            [float(x) if x is not ro.NA_Logical else None for x in ro.r("unlist(final_fit)")],
        )
    )

    initial_fit_measures = dict(
        zip(
            [
                "cfi",
                "cfi.scaled",
                "tli",
                "tli.scaled",
                "rmsea",
                "rmsea.scaled",
                "srmr",
                "aic",
                "bic",
            ],
            [
                float(x) if x is not ro.NA_Logical else None
                for x in ro.r("unlist(fit_hist[[1]])")
            ],
        )
    )

    hist_list: List[Dict[str, float]] = []
    n_hist = int(ro.r("length(fit_hist)")[0])
    for i in range(n_hist):
        vals = list(ro.r(f"unlist(fit_hist[[{i+1}]])"))
        hist_list.append(
            dict(
                zip(
                    [
                        "cfi",
                        "cfi.scaled",
                        "tli",
                        "tli.scaled",
                        "rmsea",
                        "rmsea.scaled",
                        "srmr",
                        "aic",
                        "bic",
                    ],
                    [float(x) if x is not ro.NA_Logical else None for x in vals],
                )
            )
        )

    added_list: List[Dict[str, Any]] = []
    n_added = int(ro.r("length(added)")[0])
    for i in range(n_added):
        lhs = str(ro.r(f"added[[{i+1}]]$lhs")[0])
        rhs = str(ro.r(f"added[[{i+1}]]$rhs")[0])
        mi = float(ro.r(f"added[[{i+1}]]$mi")[0])
        sepc = float(ro.r(f"added[[{i+1}]]$sepc")[0])
        mi_col = str(ro.r(f"added[[{i+1}]]$mi_col")[0])
        step = int(ro.r(f"added[[{i+1}]]$step")[0])
        added_list.append(
            {
                "lhs": lhs,
                "rhs": rhs,
                "mi": mi,
                "sepc.all": sepc,
                "mi_col": mi_col,
                "step": step,
            }
        )

    final_model_string = str(ro.r("model_cur")[0])

    return {
        "final_model_string": final_model_string,
        "fit_measures": fit_measures,
        "initial_fit_measures": initial_fit_measures,
        "added_covariances": added_list,
        "fit_history": hist_list,
    }
