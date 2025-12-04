# DiagnosticFunctions.py
# Collection of diagnostic functions for harmonization assessment (pre and post)

import warnings
from collections import Counter
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from scipy.stats import chi2
import argparse
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

"""
    Collection of statistical functions to assess and visualise batch effects in tabular data.
    Functions include:
    - Cohens_D: Calculate Cohen's d effect size between batches for each feature.
    - Mahalanobis_Distance: Calculate Mahalanobis distance between batches.
    - PC_Correlations: Perform PCA and correlate top PCs with batch and covariates.
    - fit_lmm_safe: Robustly fit a Linear Mixed Model with fallbacks and diagnostics.
    - Variance_Ratios: Calculate variance ratios between batches for each feature.
    - KS_Test: Performs two-sample Kolmogorov-Smirnov test between batches for each feature.



"""

def fit_lmm_safe(df, formula_fixed, group_col='batch', reml=False,
                 min_group_n=10, var_threshold=1e-8,
                 optimizers=('lbfgs', 'bfgs', 'powell', 'cg'),
                 maxiter=400):
    """
    Robust wrapper to fit a MixedLM to df for a single feature 'y'.

    Behavior:
      - If feature variance < var_threshold, returns success=False and notes include 'low_variance_feature'.
      - If any batch count < min_group_n, falls back to OLS (notes include 'small_group_count').
      - Scales numeric covariates (center+unit-std) to stabilize optimization.
      - Tries multiple optimizers. On success returns model results and computed stats.
      - On complete failure returns fallback OLS if possible, else None results.

    Args:
        df (pd.DataFrame): DataFrame containing 'y' and covariates.
        formula_fixed (str): Patsy formula for fixed effects (excluding random effects).
        group_col (str): Column name for grouping variable (random effect).
        reml (bool): Whether to use REML for LMM fitting.
        min_group_n (int): Minimum samples per group to attempt LMM.
        var_threshold (float): Minimum variance threshold to attempt LMM.
        optimizers (tuple): Optimizers to try for LMM fitting.
        maxiter (int): Maximum iterations for optimizer.
    Returns a dict:
        {
          'success': bool,         # True if LMM fit succeeded
          'mdf': MixedLMResults or None,
          'ols': OLSResults or None,  # fallback OLS fit for fixed part (if available)
          'notes': list[str],      # short tokens describing what happened
          'stats': dict            # summary values (var_fixed, var_batch, var_resid, R2_marginal, R2_conditional, ICC, LR_stat, pval_LRT)
        }

    """
    notes = []
    n = df.shape[0]
    if 'y' not in df.columns:
        raise ValueError("df must contain column 'y'.")
    # 1) low-variance check
    if np.nanvar(df['y']) <= var_threshold:
        notes.append('low_variance_feature')
        return {'success': False, 'mdf': None, 'ols': None, 'notes': notes, 'stats': {}}

    # 2) check minimal group sizes
    group_counts = df[group_col].value_counts()
    if (group_counts < min_group_n).any():
        notes.append('small_group_count')
        # Attempt OLS on fixed effects only and return as fallback
        try:
            import patsy
            y_vec, X = patsy.dmatrices(formula_fixed, df, return_type='dataframe')
            ols_res = sm.OLS(y_vec, X).fit()
            notes.append('fallback_ols_used')
            stats = {
                'var_fixed': float(np.nanvar(np.dot(X.values, ols_res.params.values), ddof=0)),
                'var_batch': 0.0,
                'var_resid': float(ols_res.mse_resid),
                'R2_ols_fixed': float(ols_res.rsquared)
            }
            return {'success': False, 'mdf': None, 'ols': ols_res, 'notes': notes, 'stats': stats}
        except Exception as e:
            notes.append('fallback_ols_failed')
            return {'success': False, 'mdf': None, 'ols': None, 'notes': notes, 'stats': {}}

    # 3) scale numeric covariates (help optimizers)
    for col in df.columns:
        if col in (group_col, 'y'):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            m = df[col].mean()
            s = df[col].std(ddof=0)
            if s == 0:
                df[col] = df[col] - m
            else:
                df[col] = (df[col] - m) / s

    # 4) try mixed model with multiple optimizers
    md = mixedlm(formula_fixed, groups=group_col, data=df, re_formula="1")
    last_exc = None
    for opt in optimizers:
        try:
            mdf = md.fit(reml=reml, method=opt, maxiter=maxiter, disp=False)
            # check convergence attribute when available
            converged = getattr(mdf, 'converged', True)
            if not converged:
                notes.append(f'optimizer_{opt}_no_converge')
                continue

            # try to extract random variance (handle formats)
            try:
                cov_re = mdf.cov_re
                # If cov_re is a DataFrame
                cov_re_arr = np.asarray(cov_re)
                # treat tiny elements as zero
                if cov_re_arr.size > 0 and np.nanmax(np.abs(cov_re_arr)) <= 1e-10:
                    notes.append('var_batch_near_zero')
            except Exception:
                notes.append('cov_re_inspect_failed')

            # compute fixed-effects linear predictor variance
            try:
                exog = mdf.model.exog
                fe_params = mdf.fe_params.values.reshape(-1, 1)
                linpred_fixed = np.dot(exog, fe_params).ravel()
                var_fixed = float(np.nanvar(linpred_fixed, ddof=0))
            except Exception:
                var_fixed = np.nan

            # random intercept variance
            try:
                # Some statsmodels versions return a DataFrame or scalar
                try:
                    var_batch = float(mdf.cov_re.iloc[0, 0])
                except Exception:
                    var_batch = float(np.asarray(mdf.cov_re).ravel()[0])
            except Exception:
                var_batch = np.nan

            # residual variance
            try:
                var_resid = float(mdf.scale)
            except Exception:
                var_resid = np.nan

            # totals and R2-ish metrics (Nakagawa style for Gaussian)
            total_var = var_fixed + (0.0 if np.isnan(var_batch) else var_batch) + var_resid
            R2_marginal = var_fixed / total_var if total_var != 0 and not np.isnan(var_fixed) else np.nan
            R2_conditional = (var_fixed + (0.0 if np.isnan(var_batch) else var_batch)) / total_var if total_var != 0 else np.nan
            ICC = var_batch / (var_batch + var_resid) if (not np.isnan(var_batch) and (var_batch + var_resid) != 0) else np.nan 

            # LRT vs fixed-only OLS
            try:
                # build OLS fixed-only design via patsy for the same formula
                import patsy
                y_vec, X_fixed = patsy.dmatrices(formula_fixed, df, return_type='dataframe')
                ols_fixed = sm.OLS(y_vec, X_fixed).fit()
                llf_lmm = float(mdf.llf)
                llf_ols = float(ols_fixed.llf)
                LR_stat = 2.0 * (llf_lmm - llf_ols)
                pval_LRT = float(chi2.sf(LR_stat, 1)) if np.isfinite(LR_stat) else np.nan
            except Exception:
                LR_stat = np.nan
                pval_LRT = np.nan

            stats = {
                'var_fixed': var_fixed,
                'var_batch': var_batch,
                'var_resid': var_resid,
                'R2_marginal': R2_marginal,
                'R2_conditional': R2_conditional,
                'ICC': ICC,
                'LR_stat': LR_stat,
                'pval_LRT_random': pval_LRT
            }

            return {'success': True, 'mdf': mdf, 'ols': None, 'notes': notes, 'stats': stats}
        except Exception as e:
            last_exc = e
            notes.append(f'optimizer_{opt}_failed')
            continue

    # 5) All LMM attempts failed -> fallback to OLS (try to return useful stats)
    try:
        import patsy
        y_vec, X = patsy.dmatrices(formula_fixed, df, return_type='dataframe')
        ols_res = sm.OLS(y_vec, X).fit()
        stats = {
            'var_fixed': float(np.nanvar(np.dot(X.values, ols_res.params.values), ddof=0)),
            'var_batch': 0.0,
            'var_resid': float(ols_res.mse_resid),
            'R2_ols_fixed': float(ols_res.rsquared)
        }
        notes.append('all_lmm_optimizers_failed_fallback_ols')
        return {'success': False, 'mdf': None, 'ols': ols_res, 'notes': notes, 'stats': stats}
    except Exception:
        notes.append('all_lmm_and_ols_failed')
        return {'success': False, 'mdf': None, 'ols': None, 'notes': notes, 'stats': {}}

# ------------------ Diagnostic Functions ------------------
# Cohens D function calculates the effect size between two groups for each feature.
import numpy as np
from itertools import combinations
# Cohens d function calculates the effect size between two groups for each feature.
import numpy as np
from itertools import combinations

def z_score(data):
    """
    Z-score normalization of the data matrix (samples x features).
    Use median centered by default as is more robust to outliers and non-normal distributions.
    """
    median = np.median(data, axis=0)
    std = np.std(data, axis=0, ddof=1)
    zscored = (data - median) / std
    return zscored
    

def Cohens_D(Data, batch_indices, BatchNames=None):
    """
    Calculate Cohen's d for each feature between all pairs of groups.

    Parameters:
        Data (np.ndarray): Data matrix (samples x features).
        batch_indices (list or np.ndarray): Group label for each sample (can be strings).
        BatchNames (dict or list or None, optional):
            - If dict: mapping from group value -> readable name (e.g., {'A':'Batch A', 'B':'Batch B'})
            - If list/tuple: readable names in the same order as the unique groups in batch_indices
            - If None: readable names are str(group)

    Returns:
        np.ndarray: Cohen's d values, shape = (num_pairs, num_features).
        list: Pair labels, each as a tuple of (name1, name2).
    """
    if not isinstance(Data, np.ndarray) or Data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array (samples x features).")
    if not isinstance(batch_indices, (list, np.ndarray)) or np.ndim(batch_indices) != 1:
        raise ValueError("batch_indices must be a 1D list or numpy array.")
    if Data.shape[0] != len(batch_indices):
        raise ValueError("Number of samples in Data must match length of batch_indices.")

    # preserve order of first appearance (important for string labels)
    # using dict.fromkeys on the list preserves insertion order (Python 3.7+)
    batch_indices = np.array(batch_indices, dtype=object)
    unique_groups = np.array(list(dict.fromkeys(batch_indices.tolist())))

    if len(unique_groups) < 2:
        raise ValueError("At least two unique groups are required to calculate Cohen's d")

    # Build BatchNames mapping flexibly
    if BatchNames is None:
        BatchNames_map = {g: str(g) for g in unique_groups}
    elif isinstance(BatchNames, dict):
        # Use provided dict, but fall back to str(g) if a group is missing
        BatchNames_map = {g: BatchNames.get(g, str(g)) for g in unique_groups}
    elif isinstance(BatchNames, (list, tuple)):
        if len(BatchNames) != len(unique_groups):
            raise ValueError("When BatchNames is a list/tuple its length must equal the number of unique groups.")
        BatchNames_map = {g: name for g, name in zip(unique_groups, BatchNames)}
    else:
        raise ValueError("BatchNames must be a dict, list/tuple, or None.")

    pairwise_d = []
    pair_labels = []

    for g1, g2 in combinations(unique_groups, 2):
        mask1 = batch_indices == g1
        mask2 = batch_indices == g2
        data1 = Data[mask1, :]
        data2 = Data[mask2, :]

        # Means and sample std (ddof=1)
        mean1 = np.mean(data1, axis=0)
        mean2 = np.mean(data2, axis=0)
        std1 = np.std(data1, axis=0, ddof=1)
        std2 = np.std(data2, axis=0, ddof=1)

        # pooled standard deviation (Cohen's d using average SD)
        pooled_std = np.sqrt((std1 ** 2 + std2 ** 2) / 2.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            d = (mean1 - mean2) / pooled_std
            d = np.where(np.isnan(d), 0.0, d)  # replace NaNs (e.g., zero pooled std) with 0

        pairwise_d.append(d)
        pair_labels.append((BatchNames_map[g1], BatchNames_map[g2]))
    
    # Calculate Cohen's d for each batch and the overall mean (Commented out below section as likely uneeded and will clutter report)
    """overall_mean = np.mean(Data, axis=0)
    for g in unique_groups:
        mask = batch_indices == g
        data_g = Data[mask, :]

        mean_g = np.mean(data_g, axis=0)
        std_g = np.std(data_g, axis=0, ddof=1)

        pooled_std = np.sqrt((std_g ** 2 + np.var(Data, axis=0, ddof=1)) / 2.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            d = (mean_g - overall_mean) / pooled_std
            d = np.where(np.isnan(d), 0.0, d)  # replace NaNs with 0

        pairwise_d.append(d)
        pair_labels.append((BatchNames_map[g], 'Overall'))"""

    # Convert to numpy array (shape: num_features x num_pairs) and transpose
    
    return np.array(pairwise_d), pair_labels

# PC_Correlations performs PCA on data and computes Pearson correlation of the top N principal components with a batch variable.
def PC_Correlations(
    Data,
    batch,
    N_components=None,
    covariates=None,
    variable_names=None,
    *,
    enforce_min_components_for_plotting=True
):
    """
    Perform PCA and correlate top PCs with batch and optional covariates.

    Parameters
    ----------
    Data : np.ndarray (n_samples x n_features)
        Data matrix.
    batch : array-like (n_samples,)
        Batch labels (can be strings or numbers).
    N_components : int or None
        Requested number of principal components to compute/analyze.
        If None, default is 4 (but constrained by data size).
    covariates : None or array-like (n_samples x n_covariates)
        Optional covariate matrix.
    variable_names : None or list[str]
        Optional names for the variables (first element expected 'batch' if present).
    enforce_min_components_for_plotting : bool (default True)
        If True, the routine will try to ensure at least 2 components are returned
        when the dataset allows it (helps plotting PC1 vs PC2). If the data has
        fewer than 2 possible components (e.g., n_features==1), this is not possible
        and a warning will be emitted.

    Returns
    -------
    explained_variance : np.ndarray (n_components,)
        Percent variance explained by each returned component (sum <= 100).
    scores : np.ndarray (n_samples x n_components)
        PCA scores (projection of samples).
    PC_correlations : dict
        Mapping variable name -> {'correlation': array, 'p_value': array}
        where arrays have length == n_components.
    pca : sklearn.decomposition.PCA
        The fitted PCA object (useful for components_, explained_variance_ratio_, ...)
    """
    import numpy as np
    import pandas as pd
    import warnings
    from sklearn.decomposition import PCA
    from scipy.stats import pearsonr
    # --- Input checks & normalization ---
    if not isinstance(Data, np.ndarray) or Data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array (samples x features).")
    n_samples, n_features = Data.shape

    batch = np.asarray(batch)
    if batch.ndim != 1:
        raise ValueError("batch must be a 1D array-like of length n_samples.")
    if batch.shape[0] != n_samples:
        raise ValueError("Number of samples in Data must match length of batch")

    # Covariates: if provided, convert to array and validate shape
    if covariates is not None:
        covariates = np.asarray(covariates)
        if covariates.ndim != 2:
            raise ValueError("covariates must be a 2D array (n_samples x n_covariates).")
        if covariates.shape[0] != n_samples:
            raise ValueError("covariates must have same number of rows as Data (samples).")

    # Decide how many components to compute
    # Default desired = 4 if not given
    desired = 4 if N_components is None else int(N_components)
    max_possible = min(n_samples, n_features)
    if max_possible <= 0:
        raise ValueError("Data must have at least one sample and one feature.")
    # Try to enforce at least 2 components for plotting when dataset allows it
    if enforce_min_components_for_plotting and max_possible >= 2:
        desired = max(desired, 2)
    # Final number of components we will compute
    n_comp = min(desired, max_possible)
    if n_comp < desired:
        warnings.warn(
            f"Requested N_components={desired} reduced to {n_comp} due to data size "
            f"(n_samples={n_samples}, n_features={n_features})."
        )

    # --- Fit PCA ---
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(Data)   # shape (n_samples, n_comp)
    explained_variance = (pca.explained_variance_ratio_ * 100.0)  # % per PC

    # --- Prepare variables for correlation ---
    # Factorize batch robustly (works for strings / numbers / mixed)
    batch_codes, unique_batches = pd.factorize(batch)
    # convert to float for Pearson r
    batch_var = batch_codes.astype(float)

    variables = [batch_var]
    # Optional variables names (create defaults if not provided)
    if covariates is not None:
        # If covariates has column names (DataFrame), capture them
        if isinstance(covariates, pd.DataFrame):
            cov_arr = covariates.values
            cov_names = list(map(str, covariates.columns))
        else:
            cov_arr = np.asarray(covariates)
            cov_names = [f'covariate_{i+1}' for i in range(cov_arr.shape[1])]
        # extend variables with each covariate column
        variables.extend([cov_arr[:, i].astype(float) for i in range(cov_arr.shape[1])])
    else:
        cov_names = []

    # Build variable names list (first should be 'batch' unless user provided different)
    if variable_names is None:
        variable_names_out = ['batch'] + cov_names
    else:
        # Validate provided variable_names length matches number of variables
        # Accept possibility that user included 'batch' as first element
        if len(variable_names) == len(variables):
            variable_names_out = list(variable_names)
        elif len(variable_names) == len(variables) + 0 and str(variable_names[0]).lower() == "batch":
            # e.g. user provided ['batch', 'Age', 'Sex'] while variables is [batch, Age, Sex]
            variable_names_out = list(variable_names)
        elif len(variable_names) == len(variables) - 0:
            # allow case where user gave only covariate names (no 'batch')
            variable_names_out = ['batch'] + list(variable_names)
        else:
            raise ValueError(
                "variable_names length does not match number of variables (batch + covariates). "
                f"Got variable_names length {len(variable_names)} but need {len(variables)}."
            )

    # --- Compute Pearson correlations between each PC and each variable ---
    n_used_comps = scores.shape[1]
    PC_correlations = {}
    for name, var in zip(variable_names_out, variables):
        corrs = np.empty(n_used_comps, dtype=float)
        pvals = np.empty(n_used_comps, dtype=float)
        for i in range(n_used_comps):
            try:
                r, p = pearsonr(scores[:, i], var)
            except Exception:
                r, p = np.nan, np.nan
            corrs[i] = r
            pvals[i] = p
        PC_correlations[name] = {'correlation': corrs, 'p_value': pvals}

    # Return PCA object too so callers can access components_, mean_, etc.
    return explained_variance, scores, PC_correlations, pca

# MahalanobisDistance computes the Mahalanobis distance (multivariate difference between batch and global centroids)
def Mahalanobis_Distance(Data=None, batch=None, covariates=None):

    """
    Calculate the Mahalanobis distance between batches in the data.
    Takes optional covariates and returns distances between each batch pair
    both before and after regressing out covariates. Additionally provides
    distance of each batch to the overall centroid before and after residualizing
    covariates.

    Args:
        Data (np.ndarray): Data matrix where rows are samples (n) and columns are features (p).
        batch (np.ndarray): 1D array-like batch labels for each sample (length n).
        covariates (np.ndarray, optional): Covariate matrix (n x k). An intercept will be added automatically.

    Returns:
        dict: {
            "pairwise_raw": { (b1, b2): distance, ... },
            "pairwise_resid": { (b1, b2): distance, ... } or None if no covariates,
            "centroid_raw": { (b, 'global'): distance, ... },
            "centroid_resid": { (b, 'global'): distance, ... } or None if no covariates,
            "batches": list_of_unique_batches_in_order
        }

        Keys of the inner dicts are tuples like (b1, b2) for pairwise distances and (b, 'global') for
        distances to the overall centroid.

    Raises:
        ValueError: If inputs are invalid or less than two unique batches are provided.
    """
    # ---- validations ----
    if Data is None or batch is None:
        raise ValueError("Both Data and batch must be provided.")
    Data = np.asarray(Data, dtype=float)
    batch = np.asarray(batch)
    if Data.ndim != 2:
        raise ValueError("Data must be a 2D array (samples x features).")
    n, p = Data.shape
    if batch.shape[0] != n:
        raise ValueError("Batch length must match the number of rows in Data.")
    if np.isnan(Data).any():
        raise ValueError("Data contains NaNs; please impute or remove missing values first.")

    unique_batches = np.array(list(dict.fromkeys(batch.tolist())))  # stable order
    if unique_batches.size < 2:
        raise ValueError("At least two unique batches are required.")

    # Optional covariates handling
    have_covariates = covariates is not None
    if have_covariates:
        covariates = np.asarray(covariates, dtype=float)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)
        if covariates.shape[0] != n:
            raise ValueError("Covariates must have the same number of rows as Data.")
        if np.isnan(covariates).any():
            raise ValueError("Covariates contain NaNs; please clean them first.")

    # ---- helpers ----
    def _batch_means(X):
        return {b: X[batch == b].mean(axis=0) for b in unique_batches}

    def _global_mean(X):
        return X.mean(axis=0)

    def _cov_pinv(X):
        # Sample covariance (unbiased). Use pseudo-inverse for stability (singular or p>n).
        S = np.cov(X, rowvar=False, bias=False)
        return np.linalg.pinv(S)

    def _mahal_sq(diff, Sinv):
        # Quadratic form; return sqrt for distance
        return float(np.sqrt(diff @ Sinv @ diff))

    def _pairwise_and_centroid_distances(X):
        means = _batch_means(X)
        gmean = _global_mean(X)
        Sinv = _cov_pinv(X)

        # pairwise
        pw = {}
        for (b1, b2) in combinations(unique_batches, 2):
            d = means[b1] - means[b2]
            pw[(b1, b2)] = _mahal_sq(d, Sinv)

        # centroid
        cent = {}
        for b in unique_batches:
            d = means[b] - gmean
            cent[(b, "global")] = _mahal_sq(d, Sinv)

        return pw, cent

    # ---- raw distances ----
    pairwise_raw, centroid_raw = _pairwise_and_centroid_distances(Data)

    # ---- residualize (if covariates) and compute distances again ----
    if have_covariates:
        # Add intercept
        X = np.column_stack([np.ones((n, 1)), covariates])
        # Solve least squares for each feature simultaneously
        # Data ≈ X @ B  => B = (X^T X)^+ X^T Data
        B, *_ = np.linalg.lstsq(X, Data, rcond=None)
        resid = Data - X @ B
        pairwise_resid, centroid_resid = _pairwise_and_centroid_distances(resid)
    else:
        pairwise_resid, centroid_resid = None, None

    return {
        "pairwise_raw": pairwise_raw,
        "pairwise_resid": pairwise_resid,
        "centroid_raw": centroid_raw,
        "centroid_resid": centroid_resid,
        "batches": unique_batches.tolist(),
    }

# Define a function to calculate the feature-wise ratio of variance between each unique batch pair
def Variance_Ratios(data, batch, covariates=None):
    # Define a function to calculate the feature-wise ratio of variance between each unique batch pair
    import numpy as np
    import pandas as pd
    from itertools import combinations
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array (samples x features).")
    if not isinstance(batch, (list, np.ndarray)) or np.ndim(batch) !=1:
        raise ValueError("batch must be a 1D list or numpy array.")
    if data.shape[0] != len(batch):
        raise ValueError("Number of samples in Data must match length of batch")
    
    batch = np.array(batch)
    unique_batches = np.unique(batch)
    if len(unique_batches) < 2:
        raise ValueError("At least two unique batches are required to compute ratio of variance.")
    batch_data = {}
    ratio_of_variance = {}
    """If covariates are provided, remove their effects from the data using linear regression"""
    if covariates is not None:
        from numpy.linalg import inv, pinv
        # Create new array contatining batch and covariates, estimate betas to avoid multicollinearity
        # by using pseudo inverse
        X = np.column_stack([np.ones(covariates.shape[0]), pd.get_dummies(batch, drop_first=True), covariates])  # (N, C+1)
        beta = pinv(X) @ data  # (C+1, X)
        predicted = X @ beta   # (N, X)
        # Only remove the covariate effects, not the batch effects
        data = data - predicted + (X[:,1:1+len(unique_batches)-1] @ beta[1:1+len(unique_batches)-1,:])
    # Calculate variances for each feature in each batch
    for b in unique_batches:
        batch_data[b] = data[batch == b]
    for b1, b2 in combinations(unique_batches, 2):
        var1 = np.var(batch_data[b1], axis=0, ddof=1)
        var2 = np.var(batch_data[b2], axis=0, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = var1 / var2
            ratio[np.isnan(ratio)] = 0  # Replace NaNs due to division by zero
        ratio_of_variance[(b1, b2)] = ratio
    return ratio_of_variance

# Define a function to perform two-sample Kolmogorov-Smirnov test for distribution differences between
# each unique batch pair and each batch with the overall distribution
def KS_Test(data,
                batch,
                feature_names=None,
                compare_pairs=True,
                compare_to_overall_excluding_batch=True,
                min_batch_n=3,
                alpha=0.05,
                do_fdr=True):
    """
    Improved two-sample KS testing for batch effect detection.

    Changes to old code 03/11/2025:
      - When comparing a batch to "overall", the overall distribution excludes that batch
        by default (avoids dependence/bias).
      - Optional pairwise batch-vs-batch comparisons preserved.
      - Enforces a minimum sample size per group (min_batch_n).
      - Returns both D-statistics (effect-size) and p-values, plus FDR-corrected p-values (BH).
      - Returns small summary for each comparison: proportion_significant, mean_D.
      - Handles feature names.

    Args:
      data: (n_samples, n_features) numpy array
      batch: array-like of length n_samples with batch labels
      feature_names: optional list of feature names
      compare_pairs: if True, include pairwise batch-vs-batch KS tests
      compare_to_overall_excluding_batch: if True, compare batch to overall excluding that batch
                                        (recommended). If False, compares to pooled overall (original behavior).
      min_batch_n: minimum samples required in each group to run KS for that feature (default 3)
      alpha: significance threshold for summary reporting
      do_fdr: whether to compute Benjamini-Hochberg FDR-corrected p-values per comparison

    Returns:
      dict:
        - keys are tuples like (b, 'overall') or (b1, b2)
        - each value is a dict with:
            'statistic': np.array of D statistics (length n_features)
            'p_value': np.array of p-values (nan where test not run)
            'p_value_fdr': np.array of BH-corrected p-values (if do_fdr else None)
            'n_group1': array of sample counts per feature for group1 (same across features but kept for completeness)
            'n_group2': array of counts for group2
            'summary': {'prop_significant': float, 'mean_D': float}
        - 'feature_names': list of feature names
    """
    import numpy as np
    from scipy.stats import ks_2samp
    from itertools import combinations

    def benjamini_hochberg(pvals):
        """Simple BH FDR correction. pvals can contain np.nan; those are left as np.nan."""
        p = np.asarray(pvals)
        mask = ~np.isnan(p)
        p_nonan = p[mask]
        m = len(p_nonan)
        if m == 0:
            return np.full_like(p, np.nan, dtype=float)
        order = np.argsort(p_nonan)
        ranked = np.empty(m, dtype=float)
        # compute adjusted p in reverse order
        cummin = 1.0
        adj = np.empty(m, dtype=float)
        for i in range(m-1, -1, -1):
            rank = i + 1
            pval = p_nonan[order[i]]
            adj_val = min(cummin, pval * m / rank)
            cummin = adj_val
            adj[i] = adj_val
        # put back in original order
        adj_ordered = np.empty(m, dtype=float)
        adj_ordered[order] = adj
        out = np.full_like(p, np.nan, dtype=float)
        out[mask] = np.minimum(adj_ordered, 1.0)
        return out

    # ---- Validation ----
    if not hasattr(data, "ndim") or data.ndim != 2:
        raise ValueError("data must be a 2D numpy array (samples x features).")
    n_samples, n_features = data.shape
    batch = np.array(batch)
    if batch.ndim != 1 or len(batch) != n_samples:
        raise ValueError("batch must be 1D and match number of samples in data.")
    unique_batches = np.unique(batch)
    if len(unique_batches) < 2:
        raise ValueError("At least two unique batches are required.")

    if feature_names is None:
        feature_names = [f"feature_{i+1}" for i in range(n_features)]
    elif len(feature_names) != n_features:
        raise ValueError("feature_names length must match number of features.")

    ks_results = {}
    # Pre-slice batch indices to avoid repeated boolean masks
    batch_idx = {b: np.where(batch == b)[0] for b in unique_batches}

    # Helper to run KS for two index sets
    def run_ks_for_indices(idx1, idx2):
        stats = np.full(n_features, np.nan, dtype=float)
        pvals = np.full(n_features, np.nan, dtype=float)
        n1 = np.full(n_features, 0, dtype=int)
        n2 = np.full(n_features, 0, dtype=int)
        for fi in range(n_features):
            x = data[idx1, fi]
            y = data[idx2, fi]
            n1[fi] = x.size
            n2[fi] = y.size
            if x.size >= min_batch_n and y.size >= min_batch_n:
                try:
                    stat, p = ks_2samp(x, y, alternative='two-sided', mode='auto')
                    stats[fi] = stat
                    pvals[fi] = p
                except Exception:
                    stats[fi] = np.nan
                    pvals[fi] = np.nan
            else:
                # not enough samples to run KS reliably
                stats[fi] = np.nan
                pvals[fi] = np.nan
        return stats, pvals, n1, n2

    # Compare each batch to overall (pooled) or to overall excluding the batch
    for b in unique_batches:
        idx_b = batch_idx[b]
        if compare_to_overall_excluding_batch:
            # overall excluding this batch
            idx_other = np.setdiff1d(np.arange(n_samples), idx_b, assume_unique=True)
        else:
            # pooled overall includes the batch (original behavior)
            idx_other = np.arange(n_samples)
        if idx_b.size == 0 or idx_other.size == 0:
            # shouldn't happen, but guard
            stats = np.full(n_features, np.nan, dtype=float)
            pvals = np.full(n_features, np.nan, dtype=float)
            n1 = np.zeros(n_features, dtype=int)
            n2 = np.zeros(n_features, dtype=int)
        else:
            stats, pvals, n1, n2 = run_ks_for_indices(idx_b, idx_other)

        result = {
            'statistic': stats,
            'p_value': pvals,
            'p_value_fdr': None,
            'n_group1': n1,
            'n_group2': n2,
            'summary': {
                'prop_significant': float(np.sum((pvals < alpha) & ~np.isnan(pvals)) / np.sum(~np.isnan(pvals))) if np.any(~np.isnan(pvals)) else 0.0,
                'mean_D': float(np.nanmean(stats))
            }
        }
        if do_fdr:
            result['p_value_fdr'] = benjamini_hochberg(pvals)
        ks_results[(b, 'overall')] = result

    # Pairwise comparisons if requested
    if compare_pairs:
        for b1, b2 in combinations(unique_batches, 2):
            idx1 = batch_idx[b1]
            idx2 = batch_idx[b2]
            stats, pvals, n1, n2 = run_ks_for_indices(idx1, idx2)
            result = {
                'statistic': stats,
                'p_value': pvals,
                'p_value_fdr': None,
                'n_group1': n1,
                'n_group2': n2,
                'summary': {
                    'prop_significant': float(np.sum((pvals < alpha) & ~np.isnan(pvals)) / np.sum(~np.isnan(pvals))) if np.any(~np.isnan(pvals)) else 0.0,
                    'mean_D': float(np.nanmean(stats))
                }
            }
            if do_fdr:
                result['p_value_fdr'] = benjamini_hochberg(pvals)
            ks_results[(b1, b2)] = result

    ks_results['params'] = {
        'compare_pairs': compare_pairs,
        'compare_to_overall_excluding_batch': compare_to_overall_excluding_batch,
        'min_batch_n': min_batch_n,
        'alpha': alpha,
        'do_fdr': do_fdr
    }
    return ks_results

# Define a function to perform the Levene's test for variance differences between each unique batch pair
def Levene_Test(data, batch, centre = 'median'):
    # Define a function to perform the Levene's test for variance differences between each unique batch pair
    """
    Args: data
    - data: subjects x features (np.ndarray)
    - batch: subjects x 1 (np.ndarray), batch labels
    - centre: str, optional, the center to use for the test, 'median' by default. See scipy.stats.levene for options.
    Returns:
        - levene_results: dictionary with Levene's test statistic and p-value for each pair of batches
    Raises:
        - ValueError: if Data is not a 2D array or batch is not a
        1D array, or if the number of samples in Data and batch do not match
    
    """
    import numpy as np
    from scipy.stats import levene
    from itertools import combinations
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array (samples x features).")
    if not isinstance(batch, (list, np.ndarray)) or np.ndim(batch) !=1:
        raise ValueError("batch must be a 1D list or numpy array.")
    
    if data.shape[0] != len(batch):
        raise ValueError("Number of samples in Data must match length of batch")
    batch = np.array(batch)
    unique_batches = np.unique(batch)
    if len(unique_batches) < 2:
        raise ValueError("At least two unique batches are required to perform Levene's test.")
    batch_data = {}
    levene_results = {}
    # Calculate variances for each feature in each batch
    for b in unique_batches:
        batch_data[b] = data[batch == b]
    for b1, b2 in combinations(unique_batches, 2):
        p_values = []
        statistics = []
        for feature_idx in range(data.shape[1]):
            stat, p_value = levene(batch_data[b1][:, feature_idx], batch_data[b2][:, feature_idx], center=centre)
            statistics.append(stat)
            p_values.append(p_value)
        levene_results[(b1, b2)] = {
            'statistic': np.array(statistics),
            'p_value': np.array(p_values)
        }
    return levene_results

# Function to fit LMM safely with fallbacksdef fit_lmm_safe(df, formula_fixed, group_col, min_group_n=2, var_threshold=1e-8):
def Run_LMM(Data, batch, covariates=None, feature_names=None, group_col_name='batch',
                  covariate_names=None, min_group_n=2, var_threshold=1e-8):
    """
    Runs LMM diagnostics for each feature and returns (results_df, summary).
    results_df columns: feature, success, var_fixed, var_batch, var_resid, R2_marginal, R2_conditional, ICC, notes
    summary: Counter of notes + counts
    """
    import numpy as np
    import pandas as pd
    import warnings
    from statsmodels.formula.api import mixedlm, ols
    import statsmodels.api as sm
    from scipy.stats import chi2
    import matplotlib.pyplot as plt

    Data = np.asarray(Data, dtype=float)
    n, p = Data.shape
    if feature_names is None:
        feature_names = [f'feature_{i+1}' for i in range(p)]
    if len(feature_names) != p:
        raise ValueError("feature_names length mismatch.")

    # Build base DataFrame for model inputs
    df_base = pd.DataFrame({group_col_name: np.asarray(batch)})
    if covariates is not None:
        if isinstance(covariates, pd.DataFrame):
            cov_df = covariates.reset_index(drop=True)
        else:
            cov_arr = np.asarray(covariates)
            if cov_arr.ndim == 1:
                cov_df = pd.DataFrame({covariate_names[0] if covariate_names else 'cov1': cov_arr})
            else:
                names = covariate_names if covariate_names else [f'cov{i+1}' for i in range(cov_arr.shape[1])]
                cov_df = pd.DataFrame(cov_arr, columns=names)
        df_base = pd.concat([df_base, cov_df], axis=1)

    # build formula (fixed part)
    if covariates is not None and cov_df.shape[1] > 0:
        rhs = ' + '.join(list(cov_df.columns))
        formula_fixed = f"y ~ {rhs}"
    else:
        formula_fixed = "y ~ 1"

    rows = []
    notes_counter = Counter()
    for fi in range(p):
        df = df_base.copy()
        df['y'] = Data[:, fi]
        res = fit_lmm_safe(df, formula_fixed, group_col=group_col_name,
                           min_group_n=min_group_n, var_threshold=var_threshold)
        stats = res.get('stats', {})
        notes = res.get('notes', []) or []
        # ensure consistent fields
        row = {
            'feature': feature_names[fi],
            'success': bool(res.get('success', False)), # did LMM fit succeed?
            'var_fixed': stats.get('var_fixed', np.nan), # Variance explained by fixed effects
            'var_batch': stats.get('var_batch', np.nan), # Variance explained by batch random effect
            'var_resid': stats.get('var_resid', np.nan), # Residual variance
            'R2_marginal': stats.get('R2_marginal', np.nan), # Marginal R-squared (fixed effects)
            'R2_conditional': stats.get('R2_conditional', np.nan), # Conditional R-squared (fixed + random effects)
            'ICC': stats.get('ICC', np.nan), # Intraclass correlation coefficient (batch variance / total variance)
            'LR_stat': stats.get('LR_stat', np.nan),
            'pval_LRT_random': stats.get('pval_LRT_random', np.nan),
            'notes': ';'.join(notes)
        }
        rows.append(row)
        for ntag in notes:
            notes_counter[ntag] += 1
        # also tag success vs fallback
        notes_counter['succeeded_LMM' if res.get('success', False) else 'used_fallback'] += 1

    results_df = pd.DataFrame(rows)
    summary = dict(notes_counter)
    summary['n_features'] = p
    return results_df, summary

"""
------------------ CLI Help Only Setup ------------------
 Help functions are set up to provide descriptions of the available functions without executing them.
"""
# call the help functions for each diagnostic function, for example in terminal use `python DiagnosticFunctions.py -h Cohens_D`
def setup_help_only_parser():
    parser = argparse.ArgumentParser(
        prog='DiagnosticFunctions',
        description='Diagnostic function library (use -h with a function name to view its help).'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available functions')

    # Help entry for Cohens_D
    parser_cd = subparsers.add_parser(
        'Cohens_D',
        help='Compute Cohen\'s d for two datasets',
        description="""
        Computes Cohen's d effect size per feature.
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.Cohens_D.py --Data1 <data1.npy> --Data2 <data2.npy>

        Returns a list of Cohen's d values for each feature.
        Data1 and Data2 should be numpy arrays with shape (features, samples).
        Each feature's Cohen's d is calculated as (mean1 - mean2) / pooled_std,
        where pooled_std is the square root of the average of the variances of both groups

        Note: This function does not handle missing values or NaNs.
        Ensure that Data1 and Data2 are preprocessed accordingly.

        '''
    )
    # Help entry for PcaCorr
    parser_pca = subparsers.add_parser(
        'PcaCorr',
        help='Perform PCA and correlate top PCs with batch',
        description="""
        Performs PCA on data and computes correlation of top N principal components with batch variable.
        Returns Pearson correlations, explained variance, PCA scores, and PC-batch correlations.
        Optional parameter:
        --N_components (default=3): Number of PCs to analyze.
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.PcaCorr --Data <data.npy> --batch <batch.npy>
        Returns:
        - Pearson correlation coefficients for each PC with the batch variable.
        - Explained variance for each PC.
        - PCA scores for each sample.
        - Correlation of the first N_components PCs with the batch variable.'''
    )
    parser_mahalanobis = subparsers.add_parser(
        'mahalanobis_distance',
        help='Calculate Mahalanobis distance between batches',
        description="""
        Calculates Mahalanobis distance between pairs of batches in the data.
        If covariates are provided, it will regress each feature on the covariates and return residuals from which the Mahalanobis distance is calculated.
        Args:
            Data (np.ndarray): Data matrix where rows are samples and columns are features.
            batch (np.ndarray): Batch labels for each sample.
            Cov (np.ndarray, optional): Covariance matrix. If None, it will be computed from Data.
            covariates (np.ndarray, optional): Covariates to regress out from the data.
        Returns:
            dict: A dictionary with Mahalanobis distances for each pair of batches.
        Raises:
            ValueError: If less than two unique batches are provided.
        Example:
            mahalanobis_distance(Data=data, batch=batch_labels, Cov=cov_matrix, covariates=covariates)
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.mahalanobis_distance --Data <data.npy> --batch <batch.npy>
        Returns a dictionary with Mahalanobis distances for each pair of batches.
        '''
    )

    parser_variance_ratios = subparsers.add_parser(

        'Variance_Ratios',
        help='Calculate variance ratios between batches',
        description="""
        Calculates the feature-wise ratio of variance between each unique batch pair,
        optionally removing covariate effects via linear regression.
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.Variance_ratios --Data <data.npy> --batch <batch.npy>
        Returns a dictionary with variance ratios for each pair of batches.
        '''
    )
    parser_ks_test = subparsers.add_parser(
        'KS_Test',
        help='Perform KS test between batches',
        description="""
        Performs two-sample Kolmogorov-Smirnov test for distribution differences between
        each unique batch pair and each batch with the overall distribution.
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.KS_test --Data <data.npy> --batch <batch.npy>
        Returns a dictionary with KS test statistics and p-values for each pair of batches.
        '''
    )
    parser_levene_test = subparsers.add_parser(
        'Levene_Test',
        help='Perform Levene\'s test between batches',
        description="""
        Performs Levene's test for variance differences between each unique batch pair.
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.Levene_test --Data <data.npy> --batch <batch.npy>
        Returns a dictionary with Levene's test statistics and p-values for each pair of batches.
        '''
    )
    parser_run_lmm = subparsers.add_parser(
        'Run_LMM',
        help='Run linear mixed model diagnostics',
        description="""
        Runs linear mixed model diagnostics for each feature in the data,
        returning variance components, R-squared values, ICC, and fitting notes.
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.run_lmm --Data <data.npy> --batch <batch.npy>
        Returns a DataFrame with LMM diagnostics for each feature.
        '''
    )

    return parser

if __name__ == '__main__':
    parser = setup_help_only_parser()
    parser.parse_args()