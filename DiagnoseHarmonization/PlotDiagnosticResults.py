#%%
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Sequence
import numpy as np
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for Cohens D results ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""

import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Sequence
import numpy as np
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for Cohens D results ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
def Cohens_D_plot(cohens_d: np.ndarray, pair_labels: list, df: None = None) -> None:
    """
    Plots the output of pairwise Cohen's D as bar plots with histograms of the values on different axes.
    Args:
        cohens_d (np.ndarray): 2D array of Cohen's D values (num_pairs x num_features).
        pair_labels (list): List of labels for each group pair (e.g., ['Group1 + Group2']).
        df (pd.DataFrame, optional): DataFrame for future use or extension. Currently unused.
    Returns:
        None: Displays the plots.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    import pandas as pd
    # Input validation
    if df is not None:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Dataframe must be a pandas DataFrame")
        if 'CohensD' not in df.columns:
            raise ValueError("Dataframe must contain a 'CohensD' column")
    if not isinstance(cohens_d, np.ndarray):
        raise ValueError("cohens_d must be a NumPy array.")
    
    if cohens_d.ndim != 2:
        raise ValueError("cohens_d must be a 2D array (num_pairs x num_features).")
    if not isinstance(pair_labels, list) or len(pair_labels) != cohens_d.shape[0]:
        raise ValueError("pair_labels must be a list with the same length as the number of rows in cohens_d.")
    for i in range(cohens_d.shape[0]):
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 8], wspace=0.3)
        # Histogram (left)
        ax1 = fig.add_subplot(gs[0])
        ax1.hist(cohens_d[i], bins=20, orientation='horizontal', color=[0.8, 0.2, 0.2])
        ax1.set_xlabel("Frequency")
        ax1.invert_xaxis()
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        # Bar plot (right)
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        indices = np.arange(cohens_d.shape[1])
        bars = ax2.bar(indices, cohens_d[i], color=[0.2, 0.4, 0.6])
        ax2.plot(indices, cohens_d[i], 'r.')
        # Significance lines
        effect_sizes = [
            (0.2, 'Small', 'g'),
            (0.5, 'Medium', 'b'),
            (0.8, 'Large', 'r'),
            (2.0, 'Huge', 'm')
        ]
        for val, label, color in effect_sizes:
            ax2.axhline(y=val, linestyle='--', color=color, label=label)
            ax2.axhline(y=-val, linestyle='--', color=color)
        # Labels and title
        ax2.set_xlabel("Feature Index")
        ax2.set_ylabel("Cohen's d: $(\\mu_1 - \\mu_2)/\\sigma_{pooled}$")
        ax2.set_title(f"Effect Size (Cohen's d) for {pair_labels[i]}")
        ax2.grid(True)
        plt.show()
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for PCA correlation results ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""

def variance_ratio_plot(variance_ratios:  np.ndarray, pair_labels: list, df: None = None) -> None:
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
        plt.show()

def PC_corr_plot(PrincipleComponents, batch, covariates=None, variable_names=None, PC_correlations = False):
    """
    Plots the first two PCs as a scatter plot with batch indicated by color.
    parameters:
        PrincipleComponents (np.ndarray): The PCA scores (subjects x N_components).
        batch (np.ndarray): Subjects x 1, batch labels.
        covariates (np.ndarray, optional): Subjects x covariates, additional variables to correlate with PCs. Defaults to None.
        variable_names (list of str, optional): Names for the variables. Defaults to None.
    Returns:
        None: Displays the plot.
    Raises:
        ValueError: If PrincipleComponents is not a 2D array or batch is not a
        1D array, or if the number of samples in PrincipleComponents and batch do not match.

    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    # Check number of batches
    unique_batches = np.unique(batch)
    if len(unique_batches) < 2:
        raise ValueError("At least two unique batches are required")

    # iteratvely plot the first two PCs, seperated by batch
    import matplotlib.pyplot as plt

    if variable_names is None:
        if covariates is not None:
            variables = np.column_stack((batch, covariates))
            variable_names = ['Batch'] + [f'Covariate{i+1}' for i in range(covariates.shape[1])]
        else:
            variables = batch
            variable_names = [f"Batch"]  
    # Create a DataFrame for plotting

    import pandas as pd

    PC_Names = [f"PC{i+1}" for i in range(PrincipleComponents.shape[1])]
    df = pd.DataFrame(PrincipleComponents, columns=PC_Names[:PrincipleComponents.shape[1]])
    df['batch'] = batch

    if covariates is not None:
        for i in range(covariates.shape[1]):
            df[f'Covariate{i+1}'] = covariates[:, i]

    # Plotting by batch
    plt.figure(figsize=(10, 8))
    for i in range(len(unique_batches)):
        batch_data = df[df['batch'] == unique_batches[i]]
        #plt.scatter(batch_data[variable_names[0]], batch_data[variable_names[1]], label=f'Batch {unique_batches[i]}', alpha=0.6)
        # Plotting the first two PCs as a scatter plot
        plt.scatter(PrincipleComponents[batch == unique_batches[i], 0],
                    PrincipleComponents[batch == unique_batches[i], 1],
                    label=f'Batch {unique_batches[i]}', alpha=0.6)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Scatter Plot by Batch')
        plt.legend()
        plt.grid(True)
    plt.show()

    # Plotting by covariates if provided
    if covariates is not None:
        for i in range(covariates.shape[1]):
            plt.figure(figsize=(10, 8))
            # Check if covariate is continuous or categorical, as categorical may be binary, check by number of unique values
            if len(np.unique(covariates[:, i])) <= 20:  # Assuming
                # If categorical, use a scatter plot of first two PCs, with discrete colours for each category indicated in the legend
                unique_categories = np.unique(covariates[:, i])
                for category in unique_categories:
                    category_data = df[df[f'Covariate{i+1}'] == category]
                    # Plotting the first two PCs as a scatter plot by covariate category
                    plt.scatter(PrincipleComponents[category_data.index, 0],
                                PrincipleComponents[category_data.index, 1],
                                label=f'{variable_names[i+1]} = {category}', alpha=0.6)
                    
            elif np.issubdtype(covariates[:, i].dtype, np.number):  # Check if continuous
                # If continous, use a scatter plot of first two PCs, with opacity based on covariate value
                plt.scatter(PrincipleComponents[:, 0], PrincipleComponents[:, 1],
                            c=covariates[:, i], cmap='viridis', alpha=0.6, label=f'{variable_names[i+1]} {i+1}')
                plt.colorbar(label=f'{variable_names[i+1]}{i+1}')
            else:
                raise ValueError(f"Covariate {i+1} must be either continuous or categorical, got {covariates[:, i].dtype}")
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title(f'PCA Scatter Plot by Covariate {i+1}')
            plt.legend()
            plt.grid(True)
            plt.show()  

    # Calculate and plot correlations with PCs if PC_correlations is True
    if PC_correlations:
        if covariates is None:
            raise Warning("Covariates not provided proceeding with just batch correlation")
            correlations = np.corrcoef(PrincipleComponents.T, batch.T)[:PrincipleComponents.shape[1], PrincipleComponents.shape[1]:]
        else:
            # Calculate correlations between PCs, covariates and batch
            if not isinstance(covariates, np.ndarray):
                raise ValueError("Covariates must be a numpy array")
        # Combine batch, covariates and PCS into a single array for correlation
            combined_data = np.column_stack((PrincipleComponents, batch, covariates))
            # Combine names for axes
            combined_variable_names = variable_names + [f'PC{i+1}' for i in range(PrincipleComponents.shape[1])]
            # Calculate correlations
            correlations = np.corrcoef(combined_data.T)
        # Plot the correlation matrix
        import seaborn as sns
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlations, annot=True, fmt=".2f", cmap='coolwarm',
                     xticklabels=combined_variable_names, yticklabels=combined_variable_names)
        plt.title('Correlation Matrix of PCs, Batch and Covariates')
        plt.show()    

def mahalanobis_distance_plot(results: dict,
                              annotate: bool = True,
                              figsize=(14, 5),
                              cmap="viridis",
                              show: bool = True):
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
    return fig, axes

 

# %%
