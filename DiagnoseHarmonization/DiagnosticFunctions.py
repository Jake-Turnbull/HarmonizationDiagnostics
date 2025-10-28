import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

# ------------------ Diagnostic Functions ------------------
# Cohens D function calculates the effect size between two groups for each feature.

import numpy as np
from itertools import combinations
# Cohens d function calculates the effect size between two groups for each feature.
import numpy as np
from itertools import combinations

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
    
    # Calculate Cohen's d for each batch and the overall mean
    overall_mean = np.mean(Data, axis=0)
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
        pair_labels.append((BatchNames_map[g], 'Overall'))

    return np.array(pairwise_d), pair_labels

# PcaCorr performs PCA on data and computes Pearson correlation of the top N principal components with a batch variable.
def PcaCorr(Data, batch, N_components=None, covariates=None, variable_names=None):
    """
    Perform PCA and correlate top PCs with batch and optional covariates.

    Parameters:
    - Data: subjects x features (np.ndarray)
    - batch: subjects x 1 (np.ndarray), batch labels
    - N_components:  int, optional, number of principal components to analyze (default is 3)
    - covariates:  subjects x covariates (np.ndarray), optional, additional variables to correlate with PCs
    - variable_names: list of str, optional, names for the variables (default is None, will generate default names)

    Returns:
    - explained_variance: percentage of variance explained by each principal component
    - score: PCA scores (subjects x N_components)
    - PC_correlations:  dictionary with Pearson correlations of each PC with the batch and covariates

    Raises:
    - ValueError: if Data is not a 2D array or batch is not a
    1D array, or if the number of samples in Data and batch do not match.
    - ValueError: if covariates is not None and not a 2D array
    - ValueError: if variable_names is not None and does not match the number of variables
    """
    if not isinstance(Data, np.ndarray) or Data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array (samples x features).")
    if not isinstance(batch, (list, np.ndarray)) or np.ndim(batch) != 1:
        raise ValueError("batch must be a 1D list or numpy array.")
    if Data.shape[0] != len(batch):
        raise ValueError("Number of samples in Data must match length of batch")
    if N_components is None:
        N_components = 4

    # Run PCA
    pca = PCA(n_components=N_components)
    score = pca.fit_transform(Data)
    explained_variance = pca.explained_variance_ratio_ * 100

    # Combine batch and covariates
    # Check if batch is numeric, if not convert to numeric codes
    if batch.dtype.kind in {'U', 'S', 'O'}:  # string
        batch, _ = pd.factorize(batch)   
    batch = batch.astype(float)

    variables = [batch.astype(float)]
    if covariates is not None:
        variables.extend([covariates[:, i].astype(float) for i in range(covariates.shape[1])])

    # Generate default variable names if not provided
    if variable_names is None:
        variable_names = ['batch'] + [f'covariate {i+1}' for i in range(len(variables) - 1)]

    # Compute correlations
    PC_correlations = {}
    for name, var in zip(variable_names, variables):
        corrs = []
        pvals = []
        for i in range(min(N_components, score.shape[1])):
            corr, pval = pearsonr(score[:, i], var)
            corrs.append(corr)
            pvals.append(pval)
        PC_correlations[name] = {
            'correlation': np.array(corrs),
            'p_value': np.array(pvals)
        }
        return explained_variance, score, PC_correlations
# MahalanobisDistance computes the Mahalanobis distance (multivariate difference between batch and global centroids) 
def MahalanobisDistance(Data=None, batch=None, covariates=None):

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
# Mixed effect model including cross terms with batch and covariates
def mixed_effect_interactions(data,batch,covariates,variable_names):

    """
    Make mixed effect model including cross terms with batch and covariates,

    Parameters: 
        - Data: subjects x features (np.ndarray)
        - batch: subjects x 1 (np.ndarray), batch labels
        - covariates:  subjects x covariates (np.ndarray)
        - variable_names: covariates (list)
    Returns:
        - LME model results object
    Raises:
    - ValueError: if Data is not a 2D array or batch is not a
    1D array, or if the number of samples in Data and batch do not match.
    - ValueError: if covariates is not None and not a 2D array
    - ValueError: if variable_names is not None and does not match the number of variables
    """
    # Count the number of unique groups in the batch
    
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array (samples x features).")
    if not isinstance(batch, (list, np.ndarray)) or np.ndim(batch) != 1:
        raise ValueError("group_indices must be a 1D list or numpy array.")
    
    # Define the mixed effects model as Y = X*beta + e + Z*b
    # Where Y is the data, X is the design matrix, beta are the fixed effects
    # e is the residual error, Z is the random effects design matrix and b are the
    # random effects coefficients

    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
    import patsy
    import numpy as np
    import itertools
    import warnings
    warnings.filterwarnings("ignore")
    df = pd.DataFrame(data)
    df['batch'] = batch
    for i,var in enumerate(variable_names):
        df[var] = covariates[:,i]
    # Create interaction terms
    interaction_terms = []
    for var in variable_names:
        interaction_terms.append(f'batch:{var}')
    interaction_str = ' + '.join(interaction_terms)
    # Create the formula for the mixed effects model
    formula = f'Q("0") ~ batch + {" + ".join(variable_names)} + {interaction_str}'
    # Fit the mixed effects model
    model = mixedlm(formula, df, groups=df['batch'])
    result = model.fit()
    return result  
# Define a function to calculate the feature-wise ratio of variance between each unique batch pair
def Variance_ratios(data, batch, covariates=None):
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
def KS_test(data, batch,feature_names=None):
    # Define a function to perform two-sample Kolmogorov-Smirnov test for distribution differences between
    # each unique batch pair and each batch with the overall distribution
    """
    Args: data
    - data: subjects x features (np.ndarray)
    - batch: subjects x 1 (np.ndarray), batch labels
    - feature_names: list of str, optional, names for the features (default is None, will generate default names)
    Returns:
        - ks_results: dictionary with KS test statistic and p-value for each pair of batches and
        each batch with the overall distribution
    Raises:
        - ValueError: if Data is not a 2D array or batch is not a
        1D array, or if the number of samples in Data and batch do not match    
    """
    import numpy as np
    from scipy.stats import ks_2samp
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
        raise ValueError("At least two unique batches are required to perform KS test.")
    batch_data = {}
    ks_results = {}
    # Calculate variances for each feature in each batch
    for b in unique_batches:
        batch_data[b] = data[batch == b]
        # Run two-sample KS test for each batch and the overall distribution
        p_values = []
        statistics = []
        for feature_idx in range(data.shape[1]):
            stat, p_value = ks_2samp(batch_data[b][:, feature_idx], data[:, feature_idx])
            statistics.append(stat)
            p_values.append(p_value)
        # Store results as part of dictonary under label of batch and 'overall', find data by calling ks_results[(b, 'overall')]
        ks_results[(b, 'overall')] = {
            'statistic': np.array(statistics),
            'p_value': np.array(p_values)
        }
    for b1, b2 in combinations(unique_batches, 2):
        p_values = []
        statistics = []
        for feature_idx in range(data.shape[1]):
            stat, p_value = ks_2samp(batch_data[b1][:, feature_idx], batch_data[b2][:, feature_idx])
            statistics.append(stat)
            p_values.append(p_value)
        ks_results[(b1, b2)] = {
            'statistic': np.array(statistics),
            'p_value': np.array(p_values)
        }
    if feature_names is None:
        feature_names = [f'feature {i+1}' for i in range(data.shape[1])]
    ks_results['feature_names'] = feature_names

    return ks_results
# Define a function to perform the Levene's test for variance differences between each unique batch pair
def Levene_test(data, batch, centre = 'median'):
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


"""
------------------ CLI Help Only Setup ------------------
 Help functions are set up to provide descriptions of the available functions without executing them.
"""
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

    return parser

if __name__ == '__main__':
    parser = setup_help_only_parser()
    parser.parse_args()