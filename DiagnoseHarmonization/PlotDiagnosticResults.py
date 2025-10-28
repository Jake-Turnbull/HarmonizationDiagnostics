#%%


"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- TEST WRAPPER FUNCTION ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
import inspect
from functools import wraps
from typing import Any, Callable, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.figure as mfig

def _is_figure(obj) -> bool:
    return isinstance(obj, mfig.Figure)

def _normalize_figs_from_result(result: Any) -> List[Tuple[Optional[str], mfig.Figure]]:
    """Normalize many possible return shapes into a list of (caption, Figure)."""
    if result is None:
        return []
    if _is_figure(result):
        return [(None, result)]
    if isinstance(result, tuple) and len(result) >= 1 and _is_figure(result[0]):
        return [(None, result[0])]
    if isinstance(result, (list, tuple)):
        out = []
        for item in result:
            if _is_figure(item):
                out.append((None, item))
            elif isinstance(item, (list, tuple)) and len(item) >= 2 and _is_figure(item[1]):
                out.append((str(item[0]) if item[0] is not None else None, item[1]))
        return out
    if isinstance(result, dict):
        for k in ("fig", "figure", "figures"):
            if k in result:
                return _normalize_figs_from_result(result[k])
    return []

def rep_plot_wrapper(func: Callable) -> Callable:
    """
    Decorator that:
      - optionally forces show=False (if the wrapped function supports it),
      - intercepts and removes wrapper-only kwargs (rep, log_func, caption),
      - logs returned figure(s) into rep via rep.log_plot(fig, caption) if rep provided,
      - closes figures after logging to free memory.
    """
    @wraps(func)
    def _wrapper(*args, **kwargs):
        # Extract wrapper-only args and remove them from kwargs BEFORE calling func
        rep = kwargs.pop("rep", None)
        log_func = kwargs.pop("log_func", None)
        caption_kw = kwargs.pop("caption", None)

        # If function supports 'show', force show=False unless caller explicitly set it
        try:
            sig = inspect.signature(func)
            if "show" in sig.parameters and "show" not in kwargs:
                kwargs["show"] = False
        except Exception:
            pass

        # Call original function without rep/log_func/caption in kwargs
        result = func(*args, **kwargs)

        # If neither rep nor log_func provided, return the original result unchanged
        if rep is None and log_func is None:
            return result

        # Normalize any returned figures
        figs = _normalize_figs_from_result(result)
        if not figs:
            # nothing to log; return original result for backward compatibility
            return result

        # Log each figure (use caption from return value or fallback)
        for idx, (cap, fig) in enumerate(figs):
            used_caption = cap or caption_kw or f"{func.__name__} — plot {idx+1}"
            try:
                if rep is not None:
                    rep.log_plot(fig, used_caption)
                elif callable(log_func):
                    log_func(fig, used_caption)
            except Exception as e:
                # best-effort: if rep has log_text, write the error there
                try:
                    if rep is not None and hasattr(rep, "log_text"):
                        rep.log_text(f"Failed to log figure from {func.__name__}: {e}")
                except Exception:
                    pass
            finally:
                try:
                    plt.close(fig)
                except Exception:
                    pass

        # Return original result (keeps backward compatibility)
        return result

    return _wrapper

#%%

import matplotlib.pyplot as plt
from collections.abc import Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Sequence
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for Cohens D results ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Optional
import pandas as pd

def Cohens_D_plot(
    cohens_d: np.ndarray,
    pair_labels: list,
    df: Optional[pd.DataFrame] = None,
    *,
    rep = None,            # optional StatsReporter
    caption: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    # (validation code unchanged)...
    if not isinstance(cohens_d, np.ndarray):
        raise ValueError("cohens_d must be a NumPy array.")
    if cohens_d.ndim != 2:
        raise ValueError("cohens_d must be a 2D array (num_pairs x num_features).")
    if not isinstance(pair_labels, list) or len(pair_labels) != cohens_d.shape[0]:
        raise ValueError("pair_labels must be a list with the same length as cohens_d rows.")
    
    # Create one figure per pair and return a list or just create+log each inside loop:
    figs = []
    for i in range(cohens_d.shape[0]):
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 8], wspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax1.hist(cohens_d[i], bins=20, orientation='horizontal', color=[0.8, 0.2, 0.2])
        ax1.set_xlabel("Frequency")
        ax1.invert_xaxis()
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        indices = np.arange(cohens_d.shape[1])
        ax2.bar(indices, cohens_d[i], color=[0.2, 0.4, 0.6])
        ax2.plot(indices, cohens_d[i], 'r.')
        # add effect size lines...
        ax2.set_xlabel("Feature Index")
        ax2.set_ylabel("Cohen's d")
        ax2.set_title(f"Effect Size (Cohen's d) for {pair_labels[i]}")
        fig.tight_layout()

        caption_i = caption or f"Cohen's d — {pair_labels[i]}"
        if rep is not None:
            rep.log_plot(fig, caption_i)
            plt.close(fig)
        else:
            figs.append((caption_i, fig))
            if show:
                plt.show()
    # If rep used, figs list is empty; otherwise return list for caller
    return None if rep is not None else figs
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions ratio of variance ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
def variance_ratio_plot(variance_ratios:  np.ndarray, pair_labels: list,
                         df: None = None,rep = None,show: bool = False,caption: Optional[str] = None,) -> None:
    """
    Plots the explained variance ratio for each principal component as a bar plot.

    Args:
        variance_ratios (Sequence[float]): A sequence of explained variance ratios for each principal component.
    Returns:

        None: Displays plot of vario per feature and a histogram of the values on different axes.
    Raises:
        ValueError: If variance_ratios is not a sequence of numbers.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import pandas as pd
    from matplotlib.figure import Figure

    # ---- Validation ----

    if not isinstance(variance_ratios, np.ndarray):
        raise ValueError("variance_ratios must be a NumPy array.")
    if variance_ratios.ndim != 2:
        raise ValueError("variance_ratios must be a 2D array (num_pairs x num_features).")
    if not isinstance(pair_labels, list) or len(pair_labels) != variance_ratios.shape[0]:
        raise ValueError("pair_labels must be a list with the same length as the number of rows in variance_ratios.")
    
    figs = []
    for i, label in enumerate(pair_labels):
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 8], wspace=0.3)

        # Histogram (left)
        ax1 = fig.add_subplot(gs[0])
        ax1.hist(variance_ratios[i], bins=20, orientation="horizontal", color=[0.8, 0.2, 0.2])
        ax1.set_xlabel("Frequency")
        ax1.invert_xaxis()
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")

        # Bar plot (right)
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        indices = np.arange(variance_ratios.shape[1])
        ax2.plot(indices, variance_ratios[i], "b-")
        ax2.plot(indices, variance_ratios[i], "r.")

        # Labels and title
        ax2.set_xlabel("Feature Index")
        ax2.set_ylabel("Variance Ratio: $(\\sigma_1 / \\sigma_2)$")
        ax2.set_title(f"Feature wise ratio of variance between {label}")
        ax2.grid(True)

        caption_i = caption or f"Variance ratio — {pair_labels[i]}"

        if rep is not None:
            rep.log_plot(fig, caption_i)
            plt.close(fig)
        else:
            figs.append((caption_i, fig))
            if show:
                plt.show()

    return None if rep is not None else figs
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for PCA correlation results ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
@rep_plot_wrapper
def PC_corr_plot(
    PrincipleComponents,
    batch,
    covariates=None,
    variable_names=None,
    PC_correlations=False,
    *,
    show: bool = False
):
    """
    Generate multiple PCA diagnostic plots and return a list of (caption, fig).

    Improvements / behavior:
      - covariates may be a numpy array (2D), a pandas.DataFrame, or a structured numpy array.
      - If covariates has column names (DataFrame.columns or structured dtype.names), those names are used.
      - If covariates is a plain ndarray, variable_names (if provided) will be used as covariate names.
      - variable_names may optionally include 'batch' as the first element: ['batch', 'Age', 'Sex'].
      - If no covariate names are available, defaults "Covariate1", "Covariate2", ...
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    figs = []

    # Basic validation
    if not isinstance(PrincipleComponents, np.ndarray) or PrincipleComponents.ndim != 2:
        raise ValueError("PrincipleComponents must be a 2D numpy array (samples x components).")
    if not isinstance(batch, np.ndarray) or batch.ndim != 1:
        raise ValueError("batch must be a 1D numpy array.")
    if PrincipleComponents.shape[0] != len(batch):
        raise ValueError("Number of samples in PrincipleComponents and batch must match.")
    unique_batches = np.unique(batch)
    if len(unique_batches) < 2:
        raise ValueError("At least two unique batches are required.")

    # Build DataFrame of PCs
    PC_Names = [f"PC{i+1}" for i in range(PrincipleComponents.shape[1])]
    df = pd.DataFrame(PrincipleComponents, columns=PC_Names)

    # Decide batch column name (allow variable_names to include 'batch' as first element)
    batch_col_name = "batch"
    # If variable_names explicitly provided and starts with "batch", capture it as possible batch name
    if variable_names is not None and len(variable_names) > 0 and str(variable_names[0]).lower() == "batch":
        # use the exact provided first name (preserve case) as batch label
        batch_col_name = variable_names[0]
    df[batch_col_name] = batch

    # --- Handle covariates robustly and determine covariate names ---
    cov_names = []
    cov_matrix = None  # numeric matrix (n_samples x n_covariates) used for correlations/plots

    if covariates is not None:
        # If DataFrame: use its column names
        if isinstance(covariates, pd.DataFrame):
            cov_matrix = covariates.values
            cov_names = list(map(str, covariates.columns))
        # Structured numpy array with named fields
        elif isinstance(covariates, np.ndarray) and covariates.dtype.names is not None:
            cov_names = [str(n) for n in covariates.dtype.names]
            # stack named columns into a 2D array
            cov_matrix = np.vstack([covariates[name] for name in cov_names]).T
        else:
            # array-like (convert to ndarray)
            cov_matrix = np.asarray(covariates)
            if cov_matrix.ndim != 2:
                raise ValueError("covariates must be 2D (samples x num_covariates).")
            if cov_matrix.shape[0] != PrincipleComponents.shape[0]:
                raise ValueError("Number of rows in covariates must match number of samples.")

            # If variable_names provided: it may either be exactly covariate names,
            # or include 'batch' as first element followed by covariate names.
            if variable_names is not None:
                # If user included 'batch' as first element, strip it.
                if len(variable_names) == cov_matrix.shape[1] + 1 and str(variable_names[0]).lower() == "batch":
                    cov_names = [str(x) for x in variable_names[1:]]
                elif len(variable_names) == cov_matrix.shape[1]:
                    cov_names = [str(x) for x in variable_names]
                else:
                    # inconsistent lengths: raise helpful error
                    raise ValueError(
                        "variable_names length does not match number of covariates.\n"
                        f"covariates has {cov_matrix.shape[1]} columns, "
                        f"but variable_names has length {len(variable_names)}.\n"
                        "If you include 'batch' in variable_names, put it first (e.g. ['batch', 'Age', 'Sex'])."
                    )
            else:
                # No variable_names: create defaults
                cov_names = [f"Covariate{i+1}" for i in range(cov_matrix.shape[1])]

        # Finally, assign covariate columns to df using cov_names
        # (if we reached here cov_matrix and cov_names should be set)
        if cov_matrix is None:
            raise ValueError("Unable to interpret covariates input; please supply a DataFrame, structured array, or 2D ndarray.")
        # Double-check shapes
        if cov_matrix.shape[0] != PrincipleComponents.shape[0]:
            raise ValueError("Number of rows in covariates must match number of samples.")
        if cov_matrix.shape[1] != len(cov_names):
            # defensive: if Pandas columns count mismatch (shouldn't happen), regenerate names
            cov_names = [f"Covariate{i+1}" for i in range(cov_matrix.shape[1])]

        for i, name in enumerate(cov_names):
            df[name] = cov_matrix[:, i]
    else:
        # No covariates present; ensure variable_names is either None or only contains 'batch'
        if variable_names is not None:
            if not (len(variable_names) == 1 and str(variable_names[0]).lower() == "batch"):
                raise ValueError("variable_names provided but covariates is None. Provide covariates or remove variable_names.")
        cov_names = []

    # --- 1) PCA scatter by batch ---
    fig1, ax = plt.subplots(figsize=(8, 6))
    for b in unique_batches:
        ax.scatter(df.loc[df[batch_col_name] == b, "PC1"], df.loc[df[batch_col_name] == b, "PC2"], label=f"{batch_col_name} {b}", alpha=0.7)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA Scatter Plot by Batch")
    ax.legend()
    ax.grid(True)
    figs.append(("PCA scatter by batch", fig1))

    # --- 2) PCA scatter by each covariate (if present) ---
    if cov_names:
        for name in cov_names:
            vals = df[name].values
            fig, ax = plt.subplots(figsize=(8, 6))
            # treat small-unique-count as categorical
            if len(np.unique(vals)) <= 20:
                for cat in np.unique(vals):
                    sel = df[name] == cat
                    ax.scatter(df.loc[sel, "PC1"], df.loc[sel, "PC2"], label=f"{name}={cat}", alpha=0.6)
            else:
                sc = ax.scatter(df["PC1"], df["PC2"], c=vals, cmap="viridis", alpha=0.7)
                plt.colorbar(sc, ax=ax, label=name)
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title(f"PCA Scatter Plot by {name}")
            # legend can be large; show only for categorical
            if len(np.unique(vals)) <= 20:
                ax.legend(loc="best", fontsize="small", frameon=True)
            ax.grid(True)
            figs.append((f"PCA scatter by {name}", fig))

    # --- 3) Correlation heatmap if requested ---
    if PC_correlations:
        # create combined_data and combined_names in the same order used for corr matrix
        if cov_names:
            combined_data = np.column_stack((PrincipleComponents, df[batch_col_name].values.reshape(-1, 1), df[cov_names].values))
            combined_names = PC_Names + [batch_col_name] + cov_names
        else:
            combined_data = np.column_stack((PrincipleComponents, df[batch_col_name].values.reshape(-1, 1)))
            combined_names = PC_Names + [batch_col_name]

        corr = np.corrcoef(combined_data.T)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=combined_names, yticklabels=combined_names, ax=ax)
        ax.set_title("Correlation Matrix of PCs, Batch, and Covariates")
        figs.append(("PCA correlation matrix", fig))

    # show only if requested
    if show:
        for _, f in figs:
            try:
                f.show()
            except Exception:
                # some backends may not support show on Figure objects; ignore safely
                pass

    return figs

"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for Mahalanobis distance ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
def mahalanobis_distance_plot(results: dict,
                               rep=None,
                                 annotate: bool = True,
                                   figsize=(14,5),
                                     cmap="viridis",
                                       show: bool = False):

    """
    Plot Mahalanobis distances from (...) all on ONE figure:
      - Heatmap of pairwise RAW distances
      - Heatmap of pairwise RESIDUAL distances (if available)
      - Bar chart of centroid-to-global distances (raw vs residual)

    Args:
        results (dict): Output from MahalanobisDistance(...)
        annotate (bool): Write numeric values inside heatmap cells/bars.
        figsize (tuple): Matplotlib figure size.
        cmap (str): Colormap for heatmaps.
        show (bool): If True, plt.show(); otherwise just return (fig, axes).

    Returns:
        (fig, axes): The matplotlib Figure and dict of axes.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # ---- Validation ----
    if not isinstance(results, dict):
        raise ValueError("results must be a dict produced by MahalanobisDistance(...)")

    req = ["pairwise_raw", "centroid_raw", "batches"]
    for k in req:
        if k not in results:
            raise ValueError(f"Missing required key '{k}' in results.")
    # Optional
    pairwise_resid = results.get("pairwise_resid", None)
    centroid_resid = results.get("centroid_resid", None)

    pairwise_raw = results["pairwise_raw"]
    centroid_raw = results["centroid_raw"]
    batches = results["batches"]
    if isinstance(batches, np.ndarray):
        batches = batches.tolist()
    n = len(batches)
    if n < 2:
        raise ValueError("Need at least two batches to plot distances.")

    # ---- Helpers ----
    def build_matrix(pw: dict) -> np.ndarray:
        M = np.full((n, n), np.nan, dtype=float)
        # Fill symmetric entries from pairwise dict keys (b1, b2)
        # Diagonal defined as 0 (distance of a batch to itself)
        for i in range(n):
            M[i, i] = 0.0
        if pw is None:
            return M
        for (b1, b2), d in pw.items():
            i = batches.index(b1)
            j = batches.index(b2)
            M[i, j] = d
            M[j, i] = d
        return M

    def centroid_array(cent: dict) -> np.ndarray:
        if cent is None:
            return None
        # keys like (b, 'global')
        return np.array([float(cent[(b, "global")]) for b in batches], dtype=float)

    def annotate_heatmap(ax, M):
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

    # ---- Data prep ----
    M_raw = build_matrix(pairwise_raw)
    M_resid = build_matrix(pairwise_resid) if pairwise_resid is not None else None

    # Use a shared color scale across heatmaps for fair comparison
    vmax_candidates = [np.nanmax(M_raw)]
    if M_resid is not None:
        vmax_candidates.append(np.nanmax(M_resid))
    vmax = np.nanmax(vmax_candidates)
    vmin = 0.0

    c_raw = centroid_array(centroid_raw)
    c_res = centroid_array(centroid_resid) if centroid_resid is not None else None

    # ---- Figure layout ----
    # If residuals exist: 3 panels (raw, resid, bars)
    # Else: 2 panels (raw, bars)
    has_resid = (pairwise_resid is not None) and (centroid_resid is not None)
    num_cols = 3 if has_resid else 2

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, num_cols, figure=fig, width_ratios=[1, 1, 0.9] if has_resid else [1, 1])

    ax_raw = fig.add_subplot(gs[0, 0])
    im_raw = ax_raw.imshow(M_raw, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_raw.set_title("Pairwise Mahalanobis (Raw)")
    ax_raw.set_xticks(range(n))
    ax_raw.set_yticks(range(n))
    ax_raw.set_xticklabels(batches, rotation=45, ha="right")
    ax_raw.set_yticklabels(batches)
    ax_raw.set_xlabel("Batch")
    ax_raw.set_ylabel("Batch")
    if annotate:
        annotate_heatmap(ax_raw,M_raw)

    if has_resid:
        ax_resid = fig.add_subplot(gs[0, 1])
        im_resid = ax_resid.imshow(M_resid, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_resid.set_title("Pairwise Mahalanobis (Residual)")
        ax_resid.set_xticks(range(n))
        ax_resid.set_yticks(range(n))
        ax_resid.set_xticklabels(batches, rotation=45, ha="right")
        ax_resid.set_yticklabels(batches)
        ax_resid.set_xlabel("Batch")
        ax_resid.set_ylabel("Batch")
        if annotate:
            annotate_heatmap(ax_resid,M_resid)

        # One colorbar shared by both heatmaps
        cbar = fig.colorbar(im_resid, ax=ax_raw, fraction=0.046, pad=0.2,orientation="horizontal",location="top")
        cbar = fig.colorbar(im_resid, ax=ax_resid, fraction=0.046, pad=0.2,orientation="horizontal",location="top")

        cbar.set_label("Mahalanobis distance")
    else:
        # Single colorbar for the single heatmap
        cbar = fig.colorbar(im_raw, ax=ax_raw, fraction=0.046, pad=0.04)
        cbar.set_label("Mahalanobis distance")

    # ---- Bar chart of centroid-to-global ----
    ax_bar = fig.add_subplot(gs[0, -1])
    x = np.arange(n)
    if c_res is None:
        # Only raw bars
        width = 0.6
        bars = ax_bar.bar(x, c_raw, width, label="Raw")
        ax_bar.set_title("Centroid → Global")
        if annotate:
            for b in bars:
                ax_bar.text(b.get_x() + b.get_width()/2., b.get_height(),
                            f"{b.get_height():.2f}",
                            ha='center', va='bottom', fontsize=8)
        ax_bar.legend()
    else:
        width = 0.38
        bars_raw = ax_bar.bar(x - width/2, c_raw, width, label="Raw")
        bars_res = ax_bar.bar(x + width/2, c_res, width, label="Residual")
        ax_bar.set_title("Centroid → Global (Raw vs Residual)")
        if annotate:
            for b in list(bars_raw) + list(bars_res):
                ax_bar.text(b.get_x() + b.get_width()/2., b.get_height(),
                            f"{b.get_height():.2f}",
                            ha='center', va='bottom', fontsize=8)
        ax_bar.legend()

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(batches, rotation=45, ha="right")
    ax_bar.set_ylabel("Mahalanobis distance")
    ax_bar.set_xlabel("Batch")

    fig.tight_layout()
    if show:
        plt.show()

    axes = {"heatmap_raw": ax_raw, "bars": ax_bar}
    if has_resid:
        axes["heatmap_resid"] = ax_resid
    fig.tight_layout()
    if rep is not None:
        rep.log_plot(fig, "Mahalanobis distances (raw vs residual)")
        plt.close(fig)
        return None, None  # or return a small marker that it was logged
    if show:
        plt.show()
    return fig, axes

"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for Two-sample Kolmogorov-Smirnov test ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
@rep_plot_wrapper
def KS_plot(ks_results):
    """
    Plot KS test results produced by KS_test().

    This version accepts either:
      - the original ks_results returned by the KS_test you posted, i.e.
        keys like (b,'overall') and (b1,b2) with values {'statistic': ..., 'p_value': ...}
      - OR a dict containing 'pairwise_ks' mapping -> {(b1,b2): (stat_array, p_array), ...}
        and 'feature_names'.

    The plotting uses the minimum p-value across features as a single representative p-value
    for each pair (so each pair produces one dot on the plot). Change to np.median or np.mean
    if you want a different summary.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.gridspec as gridspec

    # Basic validation
    if not isinstance(ks_results, dict):
        raise ValueError("ks_results must be a dictionary.")

    # Extract feature names
    if 'feature_names' in ks_results:
        feature_names = ks_results['feature_names']
    else:
        raise ValueError("ks_results must contain 'feature_names'.")

    # Build a canonical pairwise_ks mapping: (b1,b2) -> (stat_array, p_array)
    pairwise_ks = {}
    # If user already provided 'pairwise_ks' in desired format, accept it
    if 'pairwise_ks' in ks_results:
        # Expect values to be either dict{'statistic', 'p_value'} or tuple (stat, p)
        raw = ks_results['pairwise_ks']
        if not isinstance(raw, dict):
            raise ValueError("'pairwise_ks' must be a dict mapping pairs to results.")
        for pair, val in raw.items():
            if isinstance(val, dict) and 'statistic' in val and 'p_value' in val:
                pairwise_ks[pair] = (np.asarray(val['statistic']), np.asarray(val['p_value']))
            elif isinstance(val, (list, tuple)) and len(val) == 2:
                pairwise_ks[pair] = (np.asarray(val[0]), np.asarray(val[1]))
            else:
                raise ValueError("Each entry in 'pairwise_ks' must be dict{'statistic','p_value'} or (stat,p).")
    else:
        # Build from tuple keys like (b,'overall') and (b1,b2)
        for k, v in ks_results.items():
            if isinstance(k, tuple) and len(k) == 2 and k != ('feature_names',):
                # Expect v to be dict with 'statistic' and 'p_value'
                if isinstance(v, dict) and 'statistic' in v and 'p_value' in v:
                    pairwise_ks[k] = (np.asarray(v['statistic']), np.asarray(v['p_value']))
                elif isinstance(v, (list, tuple)) and len(v) == 2:
                    pairwise_ks[k] = (np.asarray(v[0]), np.asarray(v[1]))
                # else ignore other keys (like 'feature_names')

    # Now split into overall vs pairwise lists. We will summarize p-values by taking min across features.
    overall_pairs = []
    pairwise_pairs = []
    for (b1, b2), (stat_arr, p_arr) in pairwise_ks.items():
        # ensure p_arr is a 1D array of length n_features
        p_arr = np.asarray(p_arr).ravel()
        if b2 == 'overall' or b1 == 'overall':
            # treat any pair involving 'overall' as overall comparison
            overall_pairs.append(((b1, b2), stat_arr, p_arr))
        else:
            pairwise_pairs.append(((b1, b2), stat_arr, p_arr))

    # Helper to extract representative p-value (here min across features)
    def rep_p(p_array):
        if p_array.size == 0:
            return np.nan
        return float(np.nanmin(p_array))

    # Build arrays for plotting
    # Overall
    overall_labels = []
    overall_pvals = []
    for (b1, b2), stat_arr, p_arr in overall_pairs:
        # label as batch name (the non-overall entry)
        label = b1 if b2 == 'overall' else b2 if b1 == 'overall' else f"{b1} vs {b2}"
        overall_labels.append(str(label))
        overall_pvals.append(rep_p(p_arr))
    overall_pvals = np.array(overall_pvals)

    # Pairwise
    pair_labels = []
    pair_pvals = []
    for (b1, b2), stat_arr, p_arr in pairwise_pairs:
        pair_labels.append(f"{b1} vs {b2}")
        pair_pvals.append(rep_p(p_arr))
    pair_pvals = np.array(pair_pvals)

    # Sorting (handle NaNs by placing them at end)
    def sort_labels_and_vals(labels, vals):
        if vals.size == 0:
            return [], np.array([]), []
        sort_idx = np.argsort(np.nan_to_num(vals, nan=np.inf))
        sorted_vals = vals[sort_idx]
        sorted_labels = [labels[i] for i in sort_idx]
        return sorted_labels, sorted_vals, sort_idx

    s_over_labels, s_over_vals, _ = sort_labels_and_vals(overall_labels, overall_pvals)
    s_pair_labels, s_pair_vals, _ = sort_labels_and_vals(pair_labels, pair_pvals)

    # Convert to -log10, handle zeros or extremely small numbers safely
    def neglog10_safe(p_array):
        p = np.asarray(p_array, dtype=float)
        p = np.where(np.isfinite(p), p, np.nan)
        # replace zeros with a small value so -log10 doesn't blow up
        tiny = 1e-323  # smallest positive float > 0 for double
        p = np.where(p <= 0, tiny, p)
        return -np.log10(p)

    x_over = neglog10_safe(s_over_vals)
    x_pair = neglog10_safe(s_pair_vals)

    # Create the figure with two side-by-side horizontal dot plots
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 8], wspace=0.4)
    # Left: overall
    ax1 = fig.add_subplot(gs[0])
    if len(s_over_labels) > 0:
        y_over = np.arange(len(s_over_labels))
        ax1.scatter(x_over, y_over)
        ax1.set_yticks(y_over)
        ax1.set_yticklabels(s_over_labels)
        ax1.set_xlabel('-log10(p-value)')
    else:
        ax1.text(0.5, 0.5, "No 'batch vs overall' comparisons found", ha='center', va='center')
    ax1.set_title('Batch vs Overall (min p across features)')

    # thresholds lines (x positions)
    thresh05 = -np.log10(0.05)
    bonferroni_threshold = 0.05 / max(1, len(feature_names))
    thresh_bon = -np.log10(bonferroni_threshold)
    ax1.axvline(thresh05, color='r', linestyle='--', label='p=0.05')
    ax1.axvline(thresh_bon, color='g', linestyle='--', label='Bonferroni')

    # text for counts (count features across all overall pairs that are significant)
    # We'll count fraction of pairs (by representative p) passing thresholds
    if len(s_over_labels) > 0:
        num_sig_05 = np.sum(s_over_vals < 0.05)
        num_sig_bon = np.sum(s_over_vals < bonferroni_threshold)
        ax1.text(0.5, 0.05, f'Significant (p<0.05): {num_sig_05}/{len(s_over_labels)}', transform=ax1.transAxes, color='r')
        ax1.text(0.5, 0.0, f'Significant (Bonferroni): {num_sig_bon}/{len(s_over_labels)}', transform=ax1.transAxes, color='g')
    ax1.grid(True)
    ax1.legend()

    # Right: pairwise
    ax2 = fig.add_subplot(gs[1], sharey=ax1 if len(s_over_labels) == len(s_pair_labels) else None)
    if len(s_pair_labels) > 0:
        y_pair = np.arange(len(s_pair_labels))
        ax2.scatter(x_pair, y_pair)
        ax2.set_yticks(y_pair)
        ax2.set_yticklabels(s_pair_labels)
        ax2.set_xlabel('-log10(p-value)')
    else:
        ax2.text(0.5, 0.5, "No pairwise batch comparisons found", ha='center', va='center')
    ax2.set_title('Pairwise Batch Comparisons (min p across features)')
    ax2.axvline(thresh05, color='r', linestyle='--', label='p=0.05')
    ax2.axvline(thresh_bon, color='g', linestyle='--', label='Bonferroni')
    if len(s_pair_labels) > 0:
        num_sig_05_pair = np.sum(s_pair_vals < 0.05)
        num_sig_bon_pair = np.sum(s_pair_vals < bonferroni_threshold)
        ax2.text(0.6, 0.05, f'Significant (p<0.05): {num_sig_05_pair}/{len(s_pair_labels)}', transform=ax2.transAxes, color='r')
        ax2.text(0.6, 0.0, f'Significant (Bonferroni): {num_sig_bon_pair}/{len(s_pair_labels)}', transform=ax2.transAxes, color='g')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()
    return fig

"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for Mixed effects model ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
@rep_plot_wrapper
def mixed_effect_model_plot(results: dict, feature_names: list):
    """
    Plot the output of the mixed effects model as ordered plots of the -log10 p-values for each feature.
    Args:
        results (dict): Output from MixedEffectsModel(...)
        feature_names (list): List of feature names corresponding to the p-values.
    Returns:
        (fig, axes): The matplotlib Figure and dict of axes.
    """

# %%
