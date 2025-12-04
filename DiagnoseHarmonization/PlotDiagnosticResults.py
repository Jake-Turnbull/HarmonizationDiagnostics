#%%


"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- TEST WRAPPER FUNCTION ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
import inspect
from functools import wraps
from typing import Any, Callable, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.figure as mfig
from scipy import stats

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

"""
    Complementary plotting functions for the functions in DiagnosticFunctions.py

    Functions:
    - Z_Score_Plot: Plot histogram and heatmap of Z-scored data by batch.
    - Cohens_D_plot: Plot Cohen's d effect sizes with histograms.
    - variance_ratio_plot: Plot variance ratios between batches.
    - PC_corr_plot: Generate PCA diagnostic plots including scatter plots and correlation heatmaps.
    - PC_clustering_plot: K-means clustering and silhouette analysis of PCA results by batch.
    - Ks_Plot: Plot KS statistic between batches.
    - 


"""
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for Cohens D results ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Optional
import pandas as pd

@rep_plot_wrapper
def Z_Score_Plot(data, batch, probablity_distribution=False,draw_PDF=True):
    """
    Plots the median centered Z-score data as a heatmap and as a histogram of all scores.
    Re-order by batch for better visualisaion in the heatmap, also plot batch seperators on heatmap.
    Args:
        data (np.ndarray): 2D array of Z-scored data (samples x features).
    Returns:
        None: Displays plot of Z-scored data and a histogram of the values on different axes.
    """
    # Histogram of all Z-scores plotted by batch variable
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    from matplotlib.figure import Figure
    from scipy import stats

    # ---- Validation ----
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a NumPy array.")
    if data.ndim != 2:
        raise ValueError("data must be a 2D array (samples x features).")
    if not isinstance(batch, np.ndarray):
        if isinstance(batch, list):
            batch = np.array(batch)
        else:
            raise ValueError("batch must be a NumPy array or a list.")
    
    # Sort data by batch, batch can either be numeric or string labels here:
    sorted_indices = np.argsort(batch)
    sorted_data = data[sorted_indices, :]
    sorted_batch = batch[sorted_indices]
    unique_batches, batch_counts = np.unique(sorted_batch, return_counts=True)  
    # Create figure with gridspec
    fig = plt.figure(figsize=(14, 8))
    # Loop over unique batches and plot as histogram on same axis:
    ax1 = fig.add_subplot()
    if probablity_distribution==True:
        plot_type = 'density'
    else:
        plot_type = 'frequency'


    for i in np.unique(batch):
        batch_data = data[batch == i, :].flatten()

        ax1.hist(batch_data, bins=80, density=plot_type, alpha=0.5, label=str(i))
        # Draw an estimated normal distribution curve over histogram:
        if draw_PDF==True:
            mu, std = np.mean(batch_data), np.std(batch_data)
            xmin, xmax = ax1.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(x, mu, std)
            ax1.plot(x, p, linewidth=2)

    ax1.set_xlabel(plot_type)
    ax1.invert_xaxis()
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.legend(title="Batch")
    fig2 = plt.figure(figsize=(14, 8))
    ax2 = fig2.add_subplot()
    im = ax2.imshow(sorted_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax2.set_xlabel("Samples (sorted by batch)")
    ax2.set_ylabel("Features")
    ax2.set_title("Z-scored Data Heatmap")
    # Add batch seperators:
    batch_start = 0
    for count in batch_counts:
        batch_start += count
        ax2.axvline(x=batch_start - 0.5, color='black', linestyle='--', linewidth=1)
    fig.colorbar(im, ax=ax2, orientation='vertical', label='Z-score')
    figs = []
    figs.append(("Z-score histogram", fig))
    figs.append(("Z-score heatmap", fig2))
    return figs

@rep_plot_wrapper
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
        #fig.tight_layout()
        ax2.grid(True)
        # Ensure equal y-limits for fair comparison
        # Draw horizontal lines for small/medium/large effect sizes, green small, orange medium, red large
        for thresh, color, label in [ (0.2, 'green', 'Small'), (0.5, 'orange', 'Medium'), (0.8, 'red', 'Large') ]:
            ax2.axhline(y=thresh, color=color, linestyle='--', linewidth=1)
            ax2.axhline(y=-thresh, color=color, linestyle='--', linewidth=1)
            ax2.text(cohens_d.shape[1]-1, thresh, f' {label}', color=color, va='bottom', ha='right', fontsize=8)
            ax2.text(cohens_d.shape[1]-1, -thresh, f' {label}', color=color, va='top', ha='right', fontsize=8)
        # Set limits to have equal negatice/positive range around zero
        ylims = ax2.get_ylim()
        max_abs = max(abs(ylims[0]), abs(ylims[1]))
        ax2.set_ylim(-max_abs, max_abs)
        ax1.set_ylim(-max_abs, max_abs)


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
    show: bool = False,
    cluster_batches: bool = False
):
    """
    Generate multiple PCA diagnostic plots and return a list of (caption, fig).

    Improvements / behavior:
      - covariates may be a numpy array (2D), a pandas.DataFrame, or a structured numpy array.
      - If covariates has column names (DataFrame.columns or structured dtype.names), those names are used.
      - If covariates is a plain ndarray, variable_names (if provided) will be used as covariate names.
      - variable_names may optionally include 'batch' as the first element: ['batch', 'Age', 'Sex'].
      - If no covariate names are available, defaults "Covariate1", "Covariate2", ...


      K-means clustering of PCA points by batch and covariates to be added in future edit, additionally, 
      silhouette score calculation for batch also added. (Future work may add similar implementation for covariates if needed).

    Args:
        PrincipleComponents (np.ndarray): 2D array of PCA components (samples x components).
        batch (np.ndarray): 1D array of batch labels for each sample.
        covariates (Optional[Union[np.ndarray, pd.DataFrame]]): Optional covariate data.
        variable_names (Optional[List[str]]): Optional list of variable names for batch and covariates.
        PC_correlations (bool): If True, generate correlation heatmap.
        show (bool): If True, display plots immediately.
        
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
    # Change batch to numeric codes to prevent issues in plotting and calculating correlation:

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
    
    batch_numeric = pd.Categorical(batch).codes
    batch_col_code = f"{batch_col_name}_code"
    df[batch_col_code] = batch_numeric
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
            combined_data = np.column_stack((PrincipleComponents, df[batch_col_code].values.reshape(-1, 1), df[cov_names].values))
            combined_names = PC_Names + [batch_col_code] + cov_names
        else:
            combined_data = np.column_stack((PrincipleComponents, df[batch_col_code].values.reshape(-1, 1)))
            combined_names = PC_Names + [batch_col_code]

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
@rep_plot_wrapper
def pc_clustering_plot(
    PrincipleComponents,
    batch,
    covariates=None,
    variable_names=None,
    n_pcs_for_clustering=None,
    n_clusters_for_kmeans=None,
    random_state=0,
    *,
    show=False
):
    """
    NOTE TO USER: THIS FUNCTION IS A NEW ADDITION AND WAS PARTIALLY CREATED USING CHATGPT. PLEASE REVIEW CAREFULLY.
    Compute clustering diagnostics on PCA (or any embedding).

    Inputs:
      - PrincipleComponents: ndarray (n_samples x n_components)
      - batch: 1D array-like labels (length = n_samples)
      - covariates: optional (not used for clustering but kept for API parity)
      - variable_names: optional list (keeps same semantics as your other function)
      - n_pcs_for_clustering: int or None (default = min(10, n_components))
      - n_clusters_for_kmeans: int or None (default = number of unique batches)
      - random_state: int
      - show: bool -> call fig.show() if True

    Returns:
      - figs: list of (caption, matplotlib.Figure)
      - metrics: dict with silhouette, ARI, NMI, contingency table, chi2, km_labels, etc.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.preprocessing import LabelEncoder
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
    from scipy.stats import chi2_contingency

    # --- input validation & normalization (mirrors your style) ---
    if not isinstance(PrincipleComponents, np.ndarray) or PrincipleComponents.ndim != 2:
        raise ValueError("PrincipleComponents must be a 2D numpy array (samples x components).")
    if not isinstance(batch, np.ndarray):
        batch = np.asarray(batch)
    if batch.ndim != 1:
        raise ValueError("batch must be a 1D array-like.")
    if PrincipleComponents.shape[0] != batch.shape[0]:
        raise ValueError("Number of samples in PrincipleComponents and batch must match.")
    n_samples, n_components = PrincipleComponents.shape
    unique_batches = np.unique(batch)
    if len(unique_batches) < 1:
        raise ValueError("batch must contain at least one label.")

    # choose number of PCs to use for clustering diagnostics
    if n_pcs_for_clustering is None:
        n_pcs_for_clustering = min(10, n_components)
    else:
        n_pcs_for_clustering = min(int(n_pcs_for_clustering), n_components)
    X = PrincipleComponents[:, :n_pcs_for_clustering]

    # determine k for KMeans
    n_batches = len(unique_batches)
    k = n_clusters_for_kmeans or n_batches
    k = int(k)
    if not (1 <= k <= n_samples):
        raise ValueError("n_clusters_for_kmeans must be between 1 and n_samples")

    # label encode batch for metric functions
    le = LabelEncoder()
    try:
        batch_enc = le.fit_transform(batch)
    except Exception:
        batch_enc = le.fit_transform(batch.astype(str))

    figs = []
    metrics = {}

    # --- silhouette using batch as labels (if valid) ---
    if 2 <= n_batches <= (n_samples - 1):
        try:
            sil = silhouette_score(X, batch_enc)
            metrics["silhouette_using_batch"] = float(sil)
        except Exception as e:
            metrics["silhouette_using_batch"] = None
            metrics["silhouette_error"] = str(e)
    else:
        metrics["silhouette_using_batch"] = None
        metrics["silhouette_note"] = "silhouette requires 2 <= n_labels <= n_samples-1"

    # --- KMeans clustering ---
    # handle sklearn's n_init compatibility ('auto' introduced in newer sklearn)
    try:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    except TypeError:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km_labels = km.fit_predict(X)
    metrics["kmeans_n_clusters"] = int(k)
    metrics["kmeans_labels"] = km_labels

    # ARI / NMI against batch
    try:
        ari = adjusted_rand_score(batch_enc, km_labels)
        nmi = normalized_mutual_info_score(batch_enc, km_labels)
        metrics["kmeans_ari_vs_batch"] = float(ari)
        metrics["kmeans_nmi_vs_batch"] = float(nmi)
    except Exception as e:
        metrics["kmeans_ari_vs_batch"] = None
        metrics["kmeans_nmi_vs_batch"] = None
        metrics["kmeans_metrics_error"] = str(e)

    # contingency table + chi-square test
    ct = pd.crosstab(pd.Series(batch, name="batch"), pd.Series(km_labels, name="kmeans_label"))
    metrics["contingency_table_batch_vs_kmeans"] = ct
    try:
        chi2, pval, dof, expected = chi2_contingency(ct)
        metrics["chi2_vs_kmeans"] = {"chi2": float(chi2), "pvalue": float(pval), "dof": int(dof)}
    except Exception as e:
        metrics["chi2_vs_kmeans"] = {"error": str(e)}

    # --- Figures: KMeans scatter, compare vs batch, silhouette per-batch plot (if silhouette computed) ---
    # 1) KMeans clusters (PC1 vs PC2)
    fig_km, ax = plt.subplots(figsize=(8, 6))
    for lbl in np.unique(km_labels):
        sel = km_labels == lbl
        ax.scatter(X[sel, 0], X[sel, 1], label=f"k={lbl}", alpha=0.7, s=35)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"KMeans (k={k}) on first {n_pcs_for_clustering} PCs")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True)
    figs.append((f"KMeans clustering (k={k})", fig_km))

    # 2) Side-by-side comparison: colored by batch vs colored by kmeans
    fig_cmp, axs = plt.subplots(1, 2, figsize=(14, 5))
    # by batch
    for b in unique_batches:
        sel = batch == b
        axs[0].scatter(X[sel, 0], X[sel, 1], label=str(b), alpha=0.6, s=30)
    axs[0].set_title("By batch")
    axs[0].set_xlabel("PC1"); axs[0].set_ylabel("PC2"); axs[0].legend(fontsize="small")
    # by kmeans
    for lbl in np.unique(km_labels):
        sel = km_labels == lbl
        axs[1].scatter(X[sel, 0], X[sel, 1], label=f"k={lbl}", alpha=0.6, s=30)
    axs[1].set_title("By KMeans cluster")
    axs[1].set_xlabel("PC1"); axs[1].set_ylabel("PC2"); axs[1].legend(fontsize="small")
    fig_cmp.suptitle("Compare batch vs kmeans (PC1 vs PC2)")
    figs.append(("Compare batch vs kmeans", fig_cmp))

    # 3) Optional: silhouette per batch (if silhouette computed)
    if metrics.get("silhouette_using_batch") is not None:
        # compute individual sample silhouettes and average per batch
        try:
            from sklearn.metrics import silhouette_samples
            sample_sil = silhouette_samples(X, batch_enc)
            sil_by_batch = {}
            for b in unique_batches:
                sel = (batch == b)
                if sel.sum() > 0:
                    sil_by_batch[str(b)] = float(np.nanmean(sample_sil[sel]))
                else:
                    sil_by_batch[str(b)] = None
            metrics["silhouette_by_batch"] = sil_by_batch

            # plotting
            fig_sil, ax = plt.subplots(figsize=(8, 4))
            names = list(sil_by_batch.keys())
            vals = [sil_by_batch[n] if sil_by_batch[n] is not None else np.nan for n in names]
            ax.bar(names, vals)
            ax.set_ylabel("Average silhouette (per-batch)")
            ax.set_title("Average silhouette score per batch")
            figs.append(("Silhouette per batch", fig_sil))
        except Exception as e:
            metrics["silhouette_by_batch_error"] = str(e)
    # show if requested
    if show:
        for _, f in figs:
            try:
                f.show()
            except Exception:
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

    axes = {"heatmap_raw": ax_raw, "bars": ax_bar}
    if has_resid:
        axes["heatmap_resid"] = ax_resid
    #fig.tight_layout()
    if rep is not None:
        rep.log_plot(fig, "Mahalanobis distances (raw vs residual)")
        plt.close(fig)
        return None, None  # or return a small marker that it was logged
    if show:
        plt.show()
    return fig, axes

"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for Mixed effects model ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""

"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for Two-sample Kolmogorov-Smirnov test ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""

def KS_plot(ks_results: dict,
             feature_names: list = None,
               rep = None,            # optional StatsReporter
                 caption: Optional[str] = None,
                   show: bool = False) -> plt.Figure:
    """
    Plot the output of the two sample KS test as ordered plots of the -log10 p-values for each feature.

    Overall, returns two plots
        - one plot showing the pairwise KS test results for each feature as a dot plot (ordered as -log10 p-value)
        - One plot showing the batch vs whole dataset (excluding that batch), again as a dot plot ordered by -log10 p-value.

    Args:
        ks_results (dict): Output from TwoSampleKSTest(...)
            ks_results: keys are tuples like (b, 'overall') or (b1, b2)
        - each value is a dict with:
            'statistic': np.array of D statistics (length n_features)
            'p_value': np.array of p-values (nan where test not run)
            'p_value_fdr': np.array of BH-corrected p-values (if do_fdr else None)
            'n_group1': array of sample counts per feature for group1 (same across features but kept for completeness)
            'n_group2': array of counts for group2
            'summary': {'prop_significant': float, 'mean_D': float}
    Returns:
        figs (list): List of (caption, fig) tuples for each plot generated.

    """
    # ---- Validation ---- Structure of dictionary has batch vs over all and batch vs batch as keys

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.figure import Figure
    from matplotlib.pyplot import gca    
# plot batch vs overall on one plot and code with the legend:
    fig= plt.figure(figsize=(12, 6))
    figs = []
    ax = gca()

    for key in ks_results:
        if len(key) == 2 and key[1] == 'overall':
            b = key[0]
            res = ks_results[key]
            p_values = res['p_value']
            if feature_names is not None and len(feature_names) != len(p_values):
                raise ValueError("feature_names length must match number of features in ks_results.")
            n_features = len(p_values)
            indices = np.arange(n_features)
            # sort by -log10 p-value
            sorted_indices = np.argsort(-np.log10(p_values + 1e-10))  # add small value to avoid log(0)
            sorted_pvals = p_values[sorted_indices]
            sorted_features = feature_names[sorted_indices] if feature_names is not None else sorted_indices
            ax.plot(indices, -np.log10(sorted_pvals + 1e-10), '*',label=f'Batch {b} vs Overall')
    plt.xlabel("Features (ordered by -log10 p-value)")
    plt.ylabel("-log10 p-value")
    plt.title("KS Test: Batch vs Overall")
    plt.grid(True)
    plt.legend()
    sig_threshold_05 = -np.log10(0.05 / n_features)
    sig_threshold_01 = -np.log10(0.01 / n_features)
    plt.axhline(y=sig_threshold_05, color='r', linestyle='-', label='Significance Threshold (0.05 Bonferroni)')
    plt.axhline(y=sig_threshold_01, color='g', linestyle='-', label='Significance Threshold (0.01 Bonferroni)')
    figs.append((caption or "KS Test: Batch vs Overall", fig))

    # Repeat for batch vs batch on next figure:
    fig2 = plt.figure(figsize=(12, 6))
    ax2 = gca()
    for key in ks_results:
        if len(key) == 2 and key[1] != 'overall':
            b = key[0]
            res = ks_results[key]
            p_values = res['p_value']
            if feature_names is not None and len(feature_names) != len(p_values):
                raise ValueError("feature_names length must match number of features in ks_results.")
            n_features = len(p_values)
            indices = np.arange(n_features)
            # sort by -log10 p-value
            sorted_indices = np.argsort(-np.log10(p_values + 1e-10))  # add small value to avoid log(0)
            sorted_pvals = p_values[sorted_indices]
            sorted_features = feature_names[sorted_indices] if feature_names is not None else sorted_indices
            ax2.plot(indices, -np.log10(sorted_pvals + 1e-10),'.', label=f'Batch {b} vs Overall')
    plt.xlabel("Features (ordered by -log10 p-value)")
    plt.ylabel("-log10 p-value")
    plt.title("KS Test: Batch vs Batch")
    plt.grid(True)
    # Add an line to the plot to indicate significant threshold at 0.05 and 0.01 (Bonferroni corrected and uncorrected)
    sig_threshold_05 = -np.log10(0.05 / n_features)
    sig_threshold_01 = -np.log10(0.01 / n_features)
    plt.axhline(y=sig_threshold_05, color='r', linestyle='-', label='Significance Threshold (0.05 Bonferroni)')
    plt.axhline(y=sig_threshold_01, color='g', linestyle='-', label='Significance Threshold (0.01 Bonferroni)')

    plt.legend()
    figs.append((caption or "KS Test: Batch vs Batch", fig2))

    # Check if show is given, if so, display the plots
    for caption_i, fig in figs:
        if rep is not None:
            rep.log_plot(fig, caption_i)
            plt.close(fig)
        else: figs.append((caption_i, fig))
    if show:
        for _, fig in figs:
            fig.show()
    return rep if rep is not None else figs

