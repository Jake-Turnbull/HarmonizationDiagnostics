#!/usr/bin/env python3
"""
DiagnosticFunctions.py

Unified pipeline that runs:
 - WSV (within-subject variability)
 - Pairwise Spearman (with permutation option)
 - Mixed-effects diagnostics (pairwise site tests, ICC, etc.)
 - Mahalanobis (MD) site distances
 - Additive & Multiplicative batch tests (add_test, mult_test)

See help:
python DiagnosticFunctions.py -h
 """

from __future__ import annotations
import argparse
import json
import sys, os
import re
from typing import Iterable, Optional, Sequence, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import warnings
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResults
from scipy.stats import fligner, chi2, norm
from scipy.stats import rankdata
import statsmodels
# ---------------------------------------------------------
# Helpers & feature resolver (from your pasted functions)
# ---------------------------------------------------------
def resolve_features(
    data: pd.DataFrame,
    finalidplist: Optional[Iterable[str]] = None,
    start_col: Optional[str | int] = None,
    end_col: Optional[str | int] = None,
    pattern: Optional[str] = None,
):
    cols = list(data.columns)
    if finalidplist is not None:
        return list(finalidplist)
    if start_col is not None and end_col is not None:
        if isinstance(start_col, int):
            sname = cols[start_col]
        else:
            sname = start_col
        if isinstance(end_col, int):
            ename = cols[end_col]
        else:
            ename = end_col
        sidx = cols.index(sname)
        eidx = cols.index(ename)
        if sidx > eidx:
            sidx, eidx = eidx, sidx
        return cols[sidx : eidx + 1]
    if pattern is not None:
        return [c for c in cols if c.startswith(pattern)]
    raise ValueError("No features supplied. Provide finalidplist OR start_col/end_col OR pattern.")

def _build_fixed_formula_terms(fix_eff: Sequence[str], data: pd.DataFrame, do_zscore=False) -> List[str]:
    terms = []
    for v in fix_eff:
        z = f"zscore_{v}"
        if do_zscore and z in data.columns:
            use = z
        else:
            use = v
        if use in data.columns:
            terms.append(use)
    return terms

def _safe_fit_mixedlm(formula_fixed: str, data: pd.DataFrame, group: str, reml: bool = False):
    if group not in data.columns:
        raise KeyError(f"group '{group}' not in data")
    data = data.copy()
    try:
        mdl = smf.mixedlm(formula_fixed, data, groups=data[group])
        res = mdl.fit(reml=reml, method="lbfgs")
        return res
    except Exception as e:
        warnings.warn(f"MixedLM fit failed for formula '{formula_fixed}': {e}")
        raise

# ---------------------------------------------------------
# add_test (robust LRT -> Wald fallback)
# ---------------------------------------------------------
def add_test(
    data: pd.DataFrame,
    idvar: str,
    batchvar: str,
    features: Optional[Iterable] = None,
    fix_eff: Optional[Iterable[str]] = ("age", "sex"),
    ran_eff: Optional[Iterable[str]] = ("subject",),
    start_col: Optional[str | int] = None,
    end_col: Optional[str | int] = None,
    do_zscore: bool = True,
    reml: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    feature_cols = resolve_features(data, finalidplist=features, start_col=start_col, end_col=end_col, pattern="T1")
    V = len(feature_cols)
    if verbose:
        print(f"[add_test] found {V} features")
    rows = []
    for v, feat in enumerate(feature_cols, 1):
        if verbose:
            print(f"[add_test] ({v}/{V}) testing additive batch effect for feature: {feat}")
        fixed_terms = _build_fixed_formula_terms(list(fix_eff), data, do_zscore=do_zscore)
        fixed_str = " + ".join(fixed_terms) if len(fixed_terms) > 0 else "1"
        full_fixed = f"{feat} ~ {fixed_str} + C({batchvar})"
        reduced_fixed = f"{feat} ~ {fixed_str}"
        res_full = res_red = None
        try:
            res_full = _safe_fit_mixedlm(full_fixed, data, group=list(ran_eff)[0], reml=reml)
        except Exception as e:
            if verbose:
                print(f"  full fit failed for {feat}: {e}")
        try:
            res_red = _safe_fit_mixedlm(reduced_fixed, data, group=list(ran_eff)[0], reml=reml)
        except Exception as e:
            if verbose:
                print(f"  reduced fit failed for {feat}: {e}")
        LR = np.nan; df = np.nan; pval = np.nan; used = None
        if (res_full is not None) and (res_red is not None):
            try:
                llf_full = float(getattr(res_full, "llf", np.nan))
                llf_red = float(getattr(res_red, "llf", np.nan))
                if np.isfinite(llf_full) and np.isfinite(llf_red):
                    LR = 2.0 * (llf_full - llf_red)
                    try:
                        n_levels = int(pd.Categorical(data[batchvar]).nunique())
                        df = max(n_levels - 1, 1)
                    except Exception:
                        df = np.nan
                    if not np.isnan(df):
                        pval = float(1.0 - chi2.cdf(LR, df))
                    used = "LRT"
            except Exception as e:
                if verbose:
                    print(f"  LRT computation failed for {feat}: {e}")
        if not np.isfinite(pval):
            if res_full is not None:
                try:
                    pnames = list(res_full.params.index)
                    batch_param_indices = []
                    for i, pn in enumerate(pnames):
                        if ("C(" + batchvar + ")" in pn) or (f"{batchvar}[T." in pn) or (pn.startswith(f"{batchvar}_")):
                            batch_param_indices.append(i)
                    if len(batch_param_indices) == 0:
                        cats = pd.Categorical(data[batchvar]).categories
                        for i, pn in enumerate(pnames):
                            for lvl in cats:
                                if f"{lvl}" in str(pn) and (batchvar in pn or "C(" + batchvar in pn):
                                    batch_param_indices.append(i)
                                    break
                    if len(batch_param_indices) > 0:
                        beta = res_full.params.to_numpy(dtype=float)
                        cov = res_full.cov_params()
                        cov_mat = cov.to_numpy() if hasattr(cov, "to_numpy") else np.asarray(cov)
                        beta_b = beta[batch_param_indices]
                        Sigma_bb = cov_mat[np.ix_(batch_param_indices, batch_param_indices)]
                        try:
                            inv_Sigma_bb = np.linalg.inv(Sigma_bb)
                            W = float(beta_b.T @ inv_Sigma_bb @ beta_b)
                            df_w = len(beta_b)
                            p_w = float(1.0 - chi2.cdf(W, df_w))
                            LR, df, pval = W, df_w, p_w
                            used = "Wald"
                        except np.linalg.LinAlgError:
                            Sigma_bb_pinv = np.linalg.pinv(Sigma_bb)
                            W = float(beta_b.T @ Sigma_bb_pinv @ beta_b)
                            df_w = len(beta_b)
                            p_w = float(1.0 - chi2.cdf(W, df_w))
                            LR, df, pval = W, df_w, p_w
                            used = "Wald_pinv"
                except Exception as e:
                    if verbose:
                        print(f"  Wald fallback failed for {feat}: {e}")
        rows.append({"Feature": feat, "TestStat": LR, "df": df, "p-value": pval, "method": used})
    out = pd.DataFrame(rows)
    out = out.sort_values(by="TestStat", ascending=False).reset_index(drop=True)
    return out

# ---------------------------------------------------------
# mult_test (Fligner on residuals)
# ---------------------------------------------------------
def mult_test(
    data: pd.DataFrame,
    idvar: str,
    batchvar: str,
    features: Optional[Iterable] = None,
    fix_eff: Optional[Iterable[str]] = ("age", "sex"),
    ran_eff: Optional[Iterable[str]] = ("subject",),
    start_col: Optional[str | int] = None,
    end_col: Optional[str | int] = None,
    do_zscore: bool = True,
    reml: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    feature_cols = resolve_features(data=data, finalidplist=features, start_col=start_col, end_col=end_col, pattern="T1")
    V = len(feature_cols)
    if verbose:
        print(f"[mult_test] found {V} features")
    rows = []
    for v in range(V):
        feat = feature_cols[v]
        if verbose:
            print(f"[mult_test] testing multiplicative batch effect for feature {v+1}/{V}: {feat}")
        fixed_terms = _build_fixed_formula_terms(list(fix_eff), data, do_zscore=do_zscore)
        fixed_str = " + ".join(fixed_terms) if len(fixed_terms) > 0 else "1"
        full_fixed = f"{feat} ~ {fixed_str} + C({batchvar})"
        try:
            res_full = _safe_fit_mixedlm(full_fixed, data, group=list(ran_eff)[0], reml=reml)
        except Exception as e:
            rows.append({"Feature": feat, "ChiSq": np.nan, "DF": np.nan, "p-value": np.nan})
            if verbose:
                print(f"  fit failed for {feat}: {e}")
            continue
        resid = res_full.resid
        groups = [resid[pd.Categorical(data[batchvar]) == lvl] for lvl in pd.Categorical(data[batchvar]).categories]
        try:
            stat, pval = fligner(*groups, center='median')
            df = len(groups) - 1
        except Exception as e:
            stat = np.nan; pval = np.nan; df = np.nan
            if verbose:
                print(f"  fligner test failed for {feat}: {e}")
        rows.append({"Feature": feat, "ChiSq": float(stat) if not np.isnan(stat) else np.nan, "DF": df, "p-value": float(pval) if not np.isnan(pval) else np.nan})
    out = pd.DataFrame(rows)
    out = out.sort_values(by="ChiSq", ascending=False).reset_index(drop=True)
    return out

# ---------------------------------------------------------
# WSV (within-subject variability)
# ---------------------------------------------------------
def compute_wsv_table(
    data: pd.DataFrame,
    finalidplist: Optional[Iterable[str]] = None,
    subjectvar: str = None,
    start_col: Optional[str | int] = None,
    end_col: Optional[str | int] = None,
) -> pd.DataFrame:
    if subjectvar is None:
        raise ValueError("subjectvar must be provided.")
    if finalidplist is None:
        if start_col is None or end_col is None:
            raise ValueError("Either finalidplist OR both start_col and end_col must be provided.")
        cols = list(data.columns)
        if isinstance(start_col, int):
            start_col = cols[start_col]
        if isinstance(end_col, int):
            end_col = cols[end_col]
        start_idx = cols.index(start_col)
        end_idx = cols.index(end_col)
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        finalidplist = cols[start_idx : end_idx + 1]
    else:
        finalidplist = list(finalidplist)
    grouped = data.groupby(subjectvar)
    subjects = []
    out_columns = {col: [] for col in finalidplist}
    for subj, group in grouped:
        subjects.append(subj)
        for col in finalidplist:
            arr = group[col].dropna().to_numpy(dtype=float)
            n = arr.size
            if n == 0:
                out_columns[col].append(np.nan); continue
            mean_val = arr.mean()
            if mean_val == 0:
                out_columns[col].append(np.nan); continue
            if n == 2:
                val = abs(arr[0] - arr[1]) / mean_val * 100.0
            elif n > 2:
                val = arr.std(ddof=1) / mean_val * 100.0
            else:
                val = np.nan
            out_columns[col].append(float(val))
    WSVtab = pd.DataFrame({"subject": subjects})
    for col in finalidplist:
        WSVtab[col] = out_columns[col]
    return WSVtab

# ---------------------------------------------------------
# Pairwise Spearman (with permutations)
# ---------------------------------------------------------
def _force_numeric_vector(x) -> np.ndarray:
    if isinstance(x, pd.Series):
        res = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    else:
        res = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    return res.ravel()

def _pearson_corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return np.nan
    a = a.astype(float); b = b.astype(float)
    a_mean = a.mean(); b_mean = b.mean()
    a_dev = a - a_mean; b_dev = b - b_mean
    denom = np.sqrt((a_dev ** 2).sum() * (b_dev ** 2).sum())
    if denom == 0:
        return np.nan
    return float((a_dev * b_dev).sum() / denom)

def evaluate_pairwise_spearman(
    all_data: pd.DataFrame,
    subject_var: str,
    timepoint_var: str,
    finalidplist: Optional[Iterable[str]] = None,
    nPerm: int = 10000,
    start_col: Optional[str | int] = None,
    end_col: Optional[str | int] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    if finalidplist is None:
        if start_col is None or end_col is None:
            raise ValueError("Either finalidplist or both start_col and end_col must be provided.")
        cols = list(all_data.columns)
        if isinstance(start_col, int):
            start_name = cols[start_col]
        else:
            start_name = start_col
        if isinstance(end_col, int):
            end_name = cols[end_col]
        else:
            end_name = end_col
        start_idx = cols.index(start_name)
        end_idx = cols.index(end_name)
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        idp_list = cols[start_idx : end_idx + 1]
    else:
        idp_list = list(finalidplist)
    if subject_var not in all_data.columns:
        raise KeyError(f"subject column '{subject_var}' not found")
    if timepoint_var not in all_data.columns:
        raise KeyError(f"timepoint column '{timepoint_var}' not found")
    tp_series = all_data[timepoint_var].astype(str)
    tp_levels = pd.Index(tp_series).unique().tolist()
    nTP = len(tp_levels)
    rng = np.random.default_rng(seed)
    rows = []
    for ia in range(nTP - 1):
        for ib in range(ia + 1, nTP):
            tpA = tp_levels[ia]; tpB = tp_levels[ib]
            Ta = all_data[all_data[timepoint_var].astype(str) == tpA].copy()
            Tb = all_data[all_data[timepoint_var].astype(str) == tpB].copy()
            Ta_subj = Ta[subject_var].astype(str).to_numpy()
            Tb_subj = Tb[subject_var].astype(str).to_numpy()
            maskA = np.isin(Ta_subj, Tb_subj)
            if not np.any(maskA):
                for idp in idp_list:
                    rows.append({"TimeA": tpA,"TimeB": tpB,"IDP": idp,"nPairs": 0,"SpearmanRho": np.nan,"NullMeanRho": np.nan,"pValue": np.nan})
                continue
            common_subj = Ta_subj[maskA]; idxA = np.nonzero(maskA)[0]
            tb_index_map = {}
            for i, val in enumerate(Tb_subj):
                if val not in tb_index_map:
                    tb_index_map[val] = i
            idxB = np.array([tb_index_map[s] for s in common_subj], dtype=int)
            for idp in idp_list:
                xa_raw = Ta.iloc[idxA][idp] if idp in Ta.columns else pd.Series(dtype=float)
                yb_raw = Tb.iloc[idxB][idp] if idp in Tb.columns else pd.Series(dtype=float)
                xa = _force_numeric_vector(xa_raw); yb = _force_numeric_vector(yb_raw)
                good = ~(np.isnan(xa) | np.isnan(yb))
                xa = xa[good]; yb = yb[good]; nPairs = xa.size
                if nPairs < 3:
                    rows.append({"TimeA": tpA,"TimeB": tpB,"IDP": idp,"nPairs": int(nPairs),"SpearmanRho": np.nan,"NullMeanRho": np.nan,"pValue": np.nan})
                    continue
                xa_r = rankdata(xa, method="average"); yb_r = rankdata(yb, method="average")
                obs_rho = _pearson_corr_safe(xa_r, yb_r)
                null_rhos = np.empty(nPerm, dtype=float)
                for p in range(nPerm):
                    perm_idx = rng.permutation(nPairs)
                    null_rhos[p] = _pearson_corr_safe(xa_r, yb_r[perm_idx])
                pval = float(np.mean(np.abs(null_rhos) >= np.abs(obs_rho)))
                null_mean = float(np.nanmean(null_rhos))
                rows.append({"TimeA": tpA,"TimeB": tpB,"IDP": idp,"nPairs": int(nPairs),"SpearmanRho": float(obs_rho),"NullMeanRho": null_mean,"pValue": pval})
    results = pd.DataFrame(rows, columns=["TimeA","TimeB","IDP","nPairs","SpearmanRho","NullMeanRho","pValue"])
    return results

# ---------------------------------------------------------
# Mixed-model helpers & analyze_mixed_models (patched)
# ---------------------------------------------------------
def _force_categorical(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            if not pd.api.types.is_categorical_dtype(df[c]):
                df[c] = df[c].astype("category")
def _force_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                s = df[c].astype(str)
                extracted = s.str.extract(r'(\d+)$', expand=False)
                if extracted.notna().all():
                    vals = extracted.astype(float)
                    vals = vals - vals.min()
                    df[c] = vals
                else:
                    df[c] = pd.Categorical(s).codes.astype(float)
def _zscore_columns(df: pd.DataFrame, vars_to_zscore: Iterable[str]) -> None:
    for v in vars_to_zscore:
        if v not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[v]):
            mu = df[v].mean(skipna=True); sigma = df[v].std(skipna=True)
            zname = f"zscore_{v}"
            if pd.isna(sigma) or sigma == 0:
                df[zname] = 0.0
            else:
                df[zname] = (df[v] - mu) / sigma

def build_mixed_formula(
    tbl_in: pd.DataFrame,
    response_var: str,
    fix_eff: Iterable[str],
    ran_eff: Iterable[str],
    batch_vars: Iterable[str],
    force_categorical: Iterable[str] = (),
    force_numeric: Iterable[str] = (),
    zscore_vars: Iterable[str] = (),
    zscore_response: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    df = tbl_in.copy()
    fix_eff = list(fix_eff); ran_eff = list(ran_eff); batch_vars = list(batch_vars)
    force_categorical = list(force_categorical); force_numeric = list(force_numeric); zscore_vars = list(zscore_vars)
    _zscore_columns(df, zscore_vars)
    def present(name: str) -> bool:
        return name in df.columns
    def _maybe_use_zscore(v: str) -> str:
        zname = f"zscore_{v}"
        return zname if (zscore_response and zname in df.columns) else v
    lhs = _maybe_use_zscore(response_var)
    _force_categorical(df, batch_vars); _force_categorical(df, force_categorical); _force_numeric(df, force_numeric)
    fixed_terms: List[str] = []
    for v in fix_eff:
        use_name = _maybe_use_zscore(v)
        if present(use_name):
            fixed_terms.append(use_name)
    for v in batch_vars:
        use_name = _maybe_use_zscore(v)
        if present(use_name) and use_name not in fixed_terms:
            fixed_terms.append(use_name)
    seen = set(); fixed_terms = [x for x in fixed_terms if not (x in seen or seen.add(x))]
    fixed_str_with_batch = "1" if len(fixed_terms) == 0 else " + ".join(fixed_terms)
    batch_like = set(batch_vars) | {f"zscore_{b}" for b in batch_vars}
    fixed_no_batch = [t for t in fixed_terms if t not in batch_like]
    fixed_str_no_batch = "1" if len(fixed_no_batch) == 0 else " + ".join(fixed_no_batch)
    rand_terms = []
    for v in ran_eff:
        if present(v):
            if not pd.api.types.is_categorical_dtype(df[v]):
                df[v] = df[v].astype("category")
            rand_terms.append(f"(1|{v})")
    if len(rand_terms) == 0:
        formulas = [f"{lhs} ~ {fixed_str_with_batch}", f"{lhs} ~ 1", f"{lhs} ~ {fixed_str_no_batch}"]
    else:
        rand_str = " + ".join(rand_terms)
        formulas = [f"{lhs} ~ {fixed_str_with_batch} + {rand_str}", f"{lhs} ~ 1 + (1|subject)" if "subject" in df.columns else f"{lhs} ~ 1", f"{lhs} ~ {fixed_str_no_batch} + {rand_str}"]
    return df, formulas

def pairwise_site_tests(
    fit_result: MixedLMResults,
    group_var: str,
    data_frame: pd.DataFrame,
    alpha: float = 0.05,
    debug: bool = False,
) -> Tuple[int, pd.DataFrame]:
    if group_var not in data_frame.columns:
        raise KeyError(f"group var '{group_var}' not in data")
    if not pd.api.types.is_categorical_dtype(data_frame[group_var]):
        data_frame[group_var] = data_frame[group_var].astype("category")
    cats = list(data_frame[group_var].cat.categories)
    if len(cats) < 2:
        return 0, pd.DataFrame(columns=["siteA", "siteB", "p", "sig"])
    full_param_names = list(fit_result.params.index)
    exog_names = getattr(fit_result.model, "exog_names", None)
    if exog_names is None:
        exog_names = full_param_names.copy()
    if debug:
        print("PAIRWISE (WALD) DEBUG: full_param_names:", full_param_names)
        print("PAIRWISE (WALD) DEBUG: exog_names:", exog_names)
        print("PAIRWISE (WALD) DEBUG: categories:", cats)
    exog_to_idx = {name: i for i, name in enumerate(exog_names)}
    level_to_exog_idx = {}
    for lvl in cats:
        patt = f"[T.{lvl}]"
        found_exog = None
        for en in exog_names:
            if patt in en:
                found_exog = en; break
        if found_exog is None:
            for en in exog_names:
                if f"{group_var}_{lvl}" in en or en.endswith(f"_{lvl}") or re.search(rf"\b{re.escape(lvl)}\b", en):
                    found_exog = en; break
        level_to_exog_idx[lvl] = exog_to_idx[found_exog] if found_exog is not None else None
    if debug:
        print("PAIRWISE (WALD) DEBUG: level -> exog_idx mapping:", level_to_exog_idx)
    beta = fit_result.params.to_numpy(dtype=float)
    try:
        cov = fit_result.cov_params()
        cov_mat = cov.to_numpy() if hasattr(cov, "to_numpy") else np.asarray(cov)
    except Exception:
        raise RuntimeError("Could not obtain covariance matrix from fit_result.cov_params()")
    rows = []; sig_flags = []
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            a = cats[i]; b = cats[j]
            ex_idx_a = level_to_exog_idx.get(a); ex_idx_b = level_to_exog_idx.get(b)
            contrast_exog = np.zeros(len(exog_names), dtype=float)
            if ex_idx_a is not None: contrast_exog[ex_idx_a] = 1.0
            if ex_idx_b is not None: contrast_exog[ex_idx_b] = -1.0
            contrast_full = np.zeros(len(full_param_names), dtype=float)
            for k, exog_name in enumerate(exog_names):
                if exog_name in full_param_names:
                    pidx = full_param_names.index(exog_name)
                else:
                    pidx = None
                    for t_i, pname in enumerate(full_param_names):
                        if re.search(rf"\b{re.escape(exog_name)}\b", str(pname)) or re.search(rf"\b{re.escape(exog_name.split('[')[0])}\b", str(pname)):
                            pidx = t_i; break
                if pidx is not None:
                    contrast_full[pidx] = contrast_exog[k]
            if np.allclose(contrast_full, 0):
                pval = float("nan")
            else:
                est = float(np.dot(contrast_full, beta))
                var = float(contrast_full @ cov_mat @ contrast_full.T)
                if var <= 0 or np.isnan(var): pval = float("nan")
                else:
                    z = est / np.sqrt(var); pval = 2.0 * (1.0 - norm.cdf(abs(z)))
            sig = int(pval < alpha) if (not np.isnan(pval)) else 0
            rows.append({"siteA": a, "siteB": b, "p": pval, "sig": sig})
            sig_flags.append(sig)
            if debug:
                print(f"PAIRWISE (WALD) DEBUG: {a} vs {b} -> ex_idx_a={ex_idx_a}, ex_idx_b={ex_idx_b}, p={pval}, sig={sig}")
    full_tab = pd.DataFrame(rows, columns=["siteA", "siteB", "p", "sig"])
    return int(np.nansum(sig_flags)), full_tab

def _extract_numeric_coeff_scalar(res, varname: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    params = res.params; pvals = res.pvalues; conf = res.conf_int()
    candidates = [varname, f"zscore_{varname}"]; found_name = None
    for cand in candidates:
        if cand in params.index:
            found_name = cand; break
    if found_name is None:
        for pname in params.index:
            if re.search(rf"\b{re.escape(varname)}\b", str(pname)):
                found_name = pname; break
    if found_name is None:
        exog_names = getattr(res.model, "exog_names", None)
        if exog_names:
            for en in exog_names:
                if re.search(rf"\b{re.escape(varname)}\b", str(en)):
                    if en in params.index:
                        found_name = en; break
    if found_name is None:
        out[f"{varname}_est"] = np.nan; out[f"{varname}_pval"] = np.nan; out[f"{varname}_ciL"] = np.nan; out[f"{varname}_ciU"] = np.nan
        return out
    est = float(params.get(found_name, np.nan))
    pval = float(pvals.get(found_name, np.nan)) if found_name in pvals.index else np.nan
    if found_name in conf.index:
        ciL, ciU = conf.loc[found_name].values
    else:
        ciL = ciU = np.nan
    out[f"{varname}_est"] = est; out[f"{varname}_pval"] = pval; out[f"{varname}_ciL"] = ciL; out[f"{varname}_ciU"] = ciU
    return out

def analyze_mixed_models(
    data: pd.DataFrame,
    finalidplist: Optional[Iterable[str]],
    subject_var: str,
    batch_vars: Iterable[str],
    fix_eff: Iterable[str],
    ran_eff: Iterable[str],
    force_categorical: Iterable[str],
    force_numeric: Iterable[str],
    zscore_var: Iterable[str],
    do_zscore: bool = True,
    p_thr: float = 0.05,
    p_corr: int = 1,
    start_col: Optional[str | int] = None,
    end_col: Optional[str | int] = None,
    reml: bool = True,
) -> pd.DataFrame:
    df = data.copy()
    if finalidplist is None:
        if start_col is None or end_col is None:
            raise ValueError("Either finalidplist or start_col/end_col must be provided")
        cols = list(df.columns)
        if isinstance(start_col, int): sname = cols[start_col]
        else: sname = start_col
        if isinstance(end_col, int): ename = cols[end_col]
        else: ename = end_col
        sidx = cols.index(sname); eidx = cols.index(ename)
        if sidx > eidx: sidx, eidx = eidx, sidx
        idp_list = cols[sidx : eidx + 1]
    else:
        idp_list = list(finalidplist)

    outs = []
    if subject_var not in df.columns: raise KeyError(f"subject column '{subject_var}' not found")
    batch_vars = list(batch_vars)
    if len(batch_vars) == 0: raise ValueError("At least one batch var required")
    batch = batch_vars[0]

    for tmpidp in idp_list:
        cols_needed = list(ran_eff) + [batch] + list(fix_eff) + [tmpidp]
        cols_present = [c for c in cols_needed if c in df.columns]
        all_data = df[cols_present].copy()

        zscore_vars = list(zscore_var) + [tmpidp]
        all_data, formulas = build_mixed_formula(all_data, tmpidp, fix_eff, ran_eff, [batch],
                                                 force_categorical, force_numeric, zscore_vars, do_zscore)

        if batch in all_data.columns:
            all_data[batch] = all_data[batch].astype("category")
            counts = all_data[batch].value_counts()
            if len(counts) > 0:
                ref_site = counts.idxmax()
                current_cats = list(all_data[batch].cat.categories)
                if ref_site not in current_cats: ref_site = current_cats[0]
                new_categories = [ref_site] + [c for c in current_cats if c != ref_site]
                all_data[batch] = all_data[batch].cat.reorder_categories(new_categories, ordered=False)

        rowd: Dict[str, Any] = {}
        rowd["IDP"] = tmpidp.replace("_", "-"); rowd["batch"] = batch

        fixed_formula_full = formulas[0]; fixed_formula_full = re.sub(r"\s*\+\s*\(1\|[^)]+\)", "", fixed_formula_full)
        try:
            mdl1 = smf.mixedlm(fixed_formula_full, all_data, groups=all_data[ran_eff[0]])
            res1 = mdl1.fit(reml=reml, method="lbfgs")
        except Exception:
            rowd.update({"n_is_batchSig": np.nan,"anova_batches": np.nan,"Subj_Var": np.nan,"Resid_Var": np.nan,"ICC": np.nan,"WCV": np.nan})
            for v in fix_eff:
                rowd[f"{v}_est"] = np.nan; rowd[f"{v}_pval"] = np.nan; rowd[f"{v}_ciL"] = np.nan; rowd[f"{v}_ciU"] = np.nan
            outs.append(rowd); continue

        try:
            n_sig, full_tab = pairwise_site_tests(res1, batch, all_data, alpha=p_thr, debug=False)
        except Exception:
            n_sig = 0; full_tab = pd.DataFrame(columns=["siteA", "siteB", "p", "sig"])

        if p_corr == 0:
            rowd["n_is_batchSig"] = n_sig
        else:
            tmpsig = full_tab["p"].to_numpy(dtype=float)
            tmpsig_nonan = tmpsig[~np.isnan(tmpsig)]
            if len(tmpsig_nonan) > 0:
                p_corr_thr = 0.05 / len(tmpsig_nonan)
                rowd["n_is_batchSig"] = int(np.sum(tmpsig_nonan < p_corr_thr))
            else:
                rowd["n_is_batchSig"] = 0

        try:
            fe_pvals = res1.pvalues
            batch_mask = [bool(re.search(rf"{re.escape(batch)}", str(name))) for name in fe_pvals.index]
            anova_batches = int(np.sum(fe_pvals[batch_mask] < 0.05)) if any(batch_mask) else 0
        except Exception:
            anova_batches = np.nan
        rowd["anova_batches"] = anova_batches

        # --- Fit subject-only random model and extract variance with ML fallback if needed ---
        try:
            if ran_eff[0] in all_data.columns:
                # prepare subject-only formula (remove random-term syntax)
                formula2_raw = formulas[1]
                formula2_fixed = re.sub(r"\s*\+\s*\(1\|[^)]+\)", "", formula2_raw)

                mdl2 = smf.mixedlm(formula=formula2_fixed, data=all_data, groups=all_data[ran_eff[0]])
                res2 = mdl2.fit(reml=reml, method="lbfgs")

                # robust extractor for subj_var
                def _extract_subj_var(res_obj):
                    try:
                        cov_re = getattr(res_obj, "cov_re", None)
                        if cov_re is None:
                            return np.nan
                        arr = np.asarray(cov_re)
                        if arr.size == 0:
                            return np.nan
                        return float(arr.ravel()[0])
                    except Exception:
                        try:
                            return float(res_obj.cov_re.iloc[0, 0])
                        except Exception:
                            return np.nan

                subj_var = _extract_subj_var(res2)
                resid_var = float(getattr(res2, "scale", np.nan))

                # If subj_var is exactly zero or extremely small -> retry with ML (reml=False) to escape boundary
                if subj_var == 0 or (isinstance(subj_var, float) and np.isfinite(subj_var) and subj_var < 1e-12):
                    # attempt a safe ML fallback sequence (non-invasive; wrapped in try/except)
                    tried = False
                    fallback_methods = ["lbfgs", "powell", "nm"]
                    for method_try in fallback_methods:
                        try:
                            res2_try = mdl2.fit(reml=False, method=method_try, maxiter=5000)
                            subj_var_try = _extract_subj_var(res2_try)
                            resid_var_try = float(getattr(res2_try, "scale", np.nan))
                            # accept fallback if subj_var becomes finite and non-zero
                            if np.isfinite(subj_var_try) and subj_var_try > 0 and not np.isnan(resid_var_try):
                                res2 = res2_try
                                subj_var = subj_var_try
                                resid_var = resid_var_try
                                tried = True
                                break
                        except Exception:
                            # ignore and try next optimizer
                            continue
                    # if we didn't obtain a positive subj_var, leave subj_var as np.nan to avoid plotting zeros
                    if not tried:
                        if subj_var == 0:
                            subj_var = np.nan

                rowd["Subj_Var"] = subj_var
                rowd["Resid_Var"] = resid_var

                # compute ICC and WCV robustly (avoid divide-by-zero)
                try:
                    if np.isfinite(subj_var) and np.isfinite(resid_var) and subj_var > 0:
                        rowd["ICC"] = subj_var / (subj_var + resid_var)
                        rowd["WCV"] = resid_var / subj_var
                    else:
                        rowd["ICC"] = np.nan
                        rowd["WCV"] = np.nan
                except Exception:
                    rowd["ICC"] = np.nan
                    rowd["WCV"] = np.nan
            else:
                rowd["Subj_Var"] = np.nan; rowd["Resid_Var"] = np.nan; rowd["ICC"] = np.nan; rowd["WCV"] = np.nan
        except Exception:
            rowd["Subj_Var"] = np.nan; rowd["Resid_Var"] = np.nan; rowd["ICC"] = np.nan; rowd["WCV"] = np.nan

        # model 3: full without batch_vars: extract fixed effect coefficients for fix_eff
        try:
            mdl3 = smf.mixedlm(formula=re.sub(r"\s*\+\s*\(1\|[^)]+\)", "", formulas[2]), data=all_data, groups=all_data[ran_eff[0]])
            res3 = mdl3.fit(reml=reml, method="lbfgs")
            for v in fix_eff:
                pname = f"zscore_{v}" if f"zscore_{v}" in res3.params.index else v
                coeff_dict = _extract_numeric_coeff_scalar(res3, pname)
                cleaned = {k.replace(pname, v): val for k, val in coeff_dict.items()}
                rowd.update(cleaned)
        except Exception:
            for v in fix_eff:
                rowd[f"{v}_est"] = np.nan; rowd[f"{v}_pval"] = np.nan; rowd[f"{v}_ciL"] = np.nan; rowd[f"{v}_ciU"] = np.nan

        outs.append(rowd)

    if len(outs) == 0: return pd.DataFrame()
    first = outs[0]
    mdlnames = [k for k in first.keys() if k.endswith("_est") or k.endswith("_pval") or k.endswith("_ciL") or k.endswith("_ciU")]
    col_order = ["IDP", "batch", "n_is_batchSig", "anova_batches", "Subj_Var", "Resid_Var", "ICC", "WCV"] + mdlnames
    rows_df = pd.DataFrame(outs)
    for c in col_order:
        if c not in rows_df.columns:
            rows_df[c] = np.nan
    return rows_df[col_order]


# ---------------------------------------------------------
# Mahalanobis MD (numerical stability)
# ---------------------------------------------------------
def get_MD_numerically_stable(
    data: pd.DataFrame,
    batchvar: pd.Series | str,
    finalidplist: Optional[Iterable[str]] = None,
    start_col: Optional[str | int] = None,
    end_col: Optional[str | int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    if finalidplist is None:
        if start_col is None or end_col is None:
            raise ValueError("Either finalidplist or both start_col and end_col must be provided.")
        cols = list(data.columns)
        if isinstance(start_col, int):
            sname = cols[start_col]
        else:
            sname = start_col
        if isinstance(end_col, int):
            ename = cols[end_col]
        else:
            ename = end_col
        sidx = cols.index(sname); eidx = cols.index(ename)
        if sidx > eidx: sidx, eidx = eidx, sidx
        feature_cols = cols[sidx : eidx + 1]
    else:
        feature_cols = list(finalidplist)
    if isinstance(batchvar, str):
        if batchvar not in data.columns:
            raise KeyError(f"batchvar '{batchvar}' not in data columns")
        batch_ser = pd.Categorical(data[batchvar])
    else:
        batch_ser = pd.Categorical(pd.Series(batchvar))
    cats = list(batch_ser.categories)
    num_sites = len(cats); num_features = len(feature_cols)
    all_means = np.zeros((num_features, num_sites), dtype=float); tmpCov = np.zeros((num_features, num_features), dtype=float); site_counts: List[int] = []
    for i, lvl in enumerate(cats):
        mask = (batch_ser == lvl)
        site_df = data.loc[mask, feature_cols]
        site_df_clean = site_df.dropna(axis=0, how="any"); n_i = len(site_df_clean); site_counts.append(n_i)
        if n_i == 0:
            all_means[:, i] = np.nan; cov_i = np.zeros((num_features, num_features), dtype=float)
            if verbose: warnings.warn(f"Site '{lvl}' has zero retained samples after dropping NaNs.")
        else:
            mean_i = site_df_clean.to_numpy(dtype=float).mean(axis=0); all_means[:, i] = mean_i
            if n_i == 1: cov_i = np.zeros((num_features, num_features), dtype=float)
            else:
                cov_i = np.cov(site_df_clean.to_numpy(dtype=float), rowvar=False, ddof=1)
                if cov_i.ndim == 0: cov_i = cov_i.reshape((1, 1))
                elif cov_i.shape != (num_features, num_features):
                    cov_i = np.atleast_2d(cov_i)
                    if cov_i.shape != (num_features, num_features): cov_i = np.zeros((num_features, num_features), dtype=float)
        tmpCov += cov_i
    overallCov = tmpCov / float(num_sites); overallMean = np.nanmean(all_means, axis=1)
    try: cond_number = np.linalg.cond(overallCov)
    except Exception: cond_number = np.inf
    info: Dict[str, Any] = {"site_categories": cats, "site_counts": site_counts, "cond_number": cond_number, "num_retained_svals": 0, "overallCov": overallCov}
    MD = np.zeros((num_sites,), dtype=float)
    if cond_number > 1e15 or not np.isfinite(cond_number):
        if verbose: print(f"Using SVD-based pseudoinverse (condition = {cond_number:.2e})")
        U, s, Vt = np.linalg.svd(overallCov, full_matrices=False)
        eps = np.finfo(float).eps; tol = np.max(s) * max(overallCov.shape) * eps
        s_inv = np.zeros_like(s); keep = s > tol
        if keep.any(): s_inv[keep] = 1.0 / s[keep]
        overallCov_pinv = (Vt.T * s_inv) @ U.T
        num_retained = int(np.sum(keep)); info["num_retained_svals"] = num_retained
        if verbose: print(f"Retaining {num_retained} of {len(s)} singular values")
        for i in range(num_sites):
            mu_i = all_means[:, i]
            if np.any(np.isnan(mu_i)): MD[i] = np.nan; continue
            diff = mu_i - overallMean; delta = float(diff.T @ overallCov_pinv @ diff); MD[i] = float(np.sqrt(max(delta, 0.0)))
    else:
        if verbose: print(f"Using standard solver (condition = {cond_number:.2e})")
        for i in range(num_sites):
            mu_i = all_means[:, i]
            if np.any(np.isnan(mu_i)): MD[i] = np.nan; continue
            diff = mu_i - overallMean
            try:
                sol = np.linalg.solve(overallCov, diff); delta = float(diff.T @ sol); MD[i] = float(np.sqrt(max(delta, 0.0)))
            except np.linalg.LinAlgError:
                if verbose: warnings.warn("overallCov singular during solve; falling back to pseudoinverse.")
                overallCov_pinv = np.linalg.pinv(overallCov); delta = float(diff.T @ overallCov_pinv @ diff); MD[i] = float(np.sqrt(max(delta, 0.0)))
    mean_md = float(np.nanmean(MD)); MD_outs = np.concatenate([MD, np.array([mean_md], dtype=float)])
    site_labels = [str(c) for c in cats] + ["average_batch"]; mdvals = np.concatenate([MD, np.array([mean_md], dtype=float)])
    fullMDtab = pd.DataFrame({"batch": site_labels, "mdval": mdvals})
    return MD_outs, fullMDtab, info

# ---------------------------------------------------------
# CLI / Orchestration
# ---------------------------------------------------------
def _ensure_colnames(df: pd.DataFrame):
    df.columns = df.columns.str.replace("-", "_")
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description="Diagnose harmonization pipeline")
    p.add_argument("--input", "-i", required=True, help="input CSV file")
    p.add_argument("--output", "-o", required=False, default=".", help="output folder")
    p.add_argument("--subject", required=True, help="subject column name")
    # IMPORTANT: --batch is a single column name (no nargs)
    p.add_argument("--batch", required=True, help="batch / site column name (single column)")
    p.add_argument("--timevar", required=False, help="timepoint column name")
    p.add_argument("--finalidplist", required=False, help="list of IDP column names (JSON or comma-separated)")
    p.add_argument("--start_col", required=False, help="start column name (or index)")
    p.add_argument("--end_col", required=False, help="end column name (or index)")
    p.add_argument("--fixeff", required=False, nargs="+", help="list fixed effect column names")
    p.add_argument("--raneff", required=False, nargs="+", help="list random effect column names (defaults to subject)")
    p.add_argument("--zscore_fixeff", required=False, nargs="+", help="list of numeric fixed effect column names to be zscored")
    p.add_argument("--pattern", required=False, default="T1", help="prefix pattern fallback")
    p.add_argument("--run", required=False, default="all",
                   choices=["all", "wsv", "spearman", "mixed", "md", "add", "mult"],
                   help="which block to run")
    p.add_argument("--nperm", required=False, type=int, default=1000, help="permutations for spearman (default 1000)")
    p.add_argument("--debug", action="store_true")
    
    args = p.parse_args(argv)

    # load data and normalize column names
    df = pd.read_csv(args.input)
    df = _ensure_colnames(df)

    # parse finalidplist (JSON string or comma-separated)
    if args.finalidplist:
        try:
            finalidplist = json.loads(args.finalidplist)
        except Exception:
            finalidplist = args.finalidplist.split(",")
    else:
        finalidplist = None

    # start/end column resolution (string or integer)
    start_col = None if args.start_col is None else (int(args.start_col) if re.fullmatch(r"\d+", args.start_col) else args.start_col)
    end_col = None if args.end_col is None else (int(args.end_col) if re.fullmatch(r"\d+", args.end_col) else args.end_col)

    # --- resolve and create output directory (robust) ---
    outdir = args.output if args.output is not None else "."
    outdir = os.path.expanduser(outdir)       # allow ~/ paths
    outdir = os.path.abspath(outdir)          # make absolute to avoid cwd confusion
    # ensure directory exists
    os.makedirs(outdir, exist_ok=True)

    # Helpful debug: show exactly where files will be written
    print(f"Writing all outputs to directory: {outdir}")


    # --- enforce that batch is a SINGLE column name ---
    if isinstance(args.batch, (list, tuple)):
        # defensive: argparse shouldn't have produced a list, but handle gracefully
        if len(args.batch) == 0:
            raise ValueError("--batch must specify at least one column name")
        if len(args.batch) > 1:
            raise ValueError("--batch must be a single column name (you passed multiple)")
        batch_col = args.batch[0]
    else:
        batch_col = args.batch

    if batch_col not in df.columns:
        raise KeyError(f"Batch column '{batch_col}' not found in data columns: {df.columns.tolist()}")

    # Decide which time variable to use
    if args.timevar is not None:
        time_var = args.timevar
    else:
        if "timepoint" in df.columns:
            time_var = "timepoint"
        elif "time" in df.columns:
            time_var = "time"
        else:
            raise ValueError("No time variable found. Provide --timevar explicitly.")

    # Normalize fix / ran / zscore lists to safe defaults
    fix_eff = args.fixeff or []
    ran_eff = args.raneff or [args.subject]      # default random-effect is subject
    zscore_var = args.zscore_fixeff or []

    # Run requested block(s)
    if args.run in ("all", "wsv"):
        print("Running WSV...")
        wsv = compute_wsv_table(df, finalidplist=finalidplist, subjectvar=args.subject,
                                start_col=start_col, end_col=end_col)
        wsv.to_csv(os.path.join(outdir, "wsv_table.csv"), index=False)
        print("WSV saved to wsv_table.csv")

    if args.run in ("all", "spearman"):
        print("Running pairwise Spearman (may be slow)...")
        spearman = evaluate_pairwise_spearman(
            df,
            subject_var=args.subject,
            timepoint_var=time_var,
            finalidplist=finalidplist,
            nPerm=args.nperm,
            start_col=start_col,
            end_col=end_col,
            seed=0
        )
        spearman.to_csv(os.path.join(outdir, "pairwise_spearman.csv"), index=False)
        print("Spearman saved to pairwise_spearman.csv")

    if args.run in ("all", "mixed"):
        print("Running mixed-model diagnostics (this may take a while)...")
        # use single batch column but pass as a one-element list where functions expect iterables
        force_categorical = [args.subject, batch_col]
        force_numeric = []
        mixed_res = analyze_mixed_models(
            df,
            finalidplist=finalidplist,
            subject_var=args.subject,
            batch_vars=[batch_col],      # single batch column wrapped in a list
            fix_eff=fix_eff,
            ran_eff=ran_eff,
            force_categorical=force_categorical,
            force_numeric=force_numeric,
            zscore_var=zscore_var,
            do_zscore=True,
            start_col=start_col,
            end_col=end_col
        )
        mixed_res.to_csv(os.path.join(outdir, "mixed_models_results.csv"), index=False)
        print("Mixed diagnostics saved to mixed_models_results.csv")

    if args.run in ("all", "md"):
        print("Running MD computations...")
        MD_outs, MD_tab, info = get_MD_numerically_stable(
            df,
            batchvar=batch_col,                 # pass single column name (string)
            finalidplist=finalidplist,
            start_col=start_col,
            end_col=end_col,
            verbose=not args.debug
        )
        MD_tab.to_csv(os.path.join(outdir, "md_by_site.csv"), index=False)
        print("MD saved to md_by_site.csv")

    if args.run in ("all", "add"):
        print("Running additive tests...")
        add_res = add_test(
            df,
            idvar=args.subject,
            batchvar=batch_col,                # single batch column name
            features=finalidplist,
            fix_eff=fix_eff if len(fix_eff) > 0 else ["age", "sex"],
            ran_eff=ran_eff if len(ran_eff) > 0 else [args.subject],
            start_col=start_col,
            end_col=end_col,
            verbose=not args.debug
        )
        add_res.to_csv(os.path.join(outdir, "add_test.csv"), index=False)
        print("Additive tests saved to add_test.csv")

    if args.run in ("all", "mult"):
        print("Running multiplicative tests...")
        mult_res = mult_test(
            df,
            idvar=args.subject,
            batchvar=batch_col,                # single batch column name
            features=finalidplist,
            fix_eff=fix_eff if len(fix_eff) > 0 else ["age", "sex"],
            ran_eff=ran_eff if len(ran_eff) > 0 else [args.subject],
            start_col=start_col,
            end_col=end_col,
            verbose=not args.debug
        )
        mult_res.to_csv(os.path.join(outdir, "mult_test.csv"), index=False)
        print("Multiplicative tests saved to mult_test.csv")

    print("Done.")



if __name__ == "__main__":
    main()
