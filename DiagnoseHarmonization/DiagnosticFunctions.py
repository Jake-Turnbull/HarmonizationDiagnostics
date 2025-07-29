import argparse
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

# ------------------ Diagnostic Functions ------------------
# Cohens D function calculates the effect size between two groups for each feature.

import numpy as np
from itertools import combinations

def Cohens_D(Data, batch_indices, BatchNames=None):
    """
    Calculate Cohen's d for each feature between all pairs of groups.

    Parameters:
        Data (np.ndarray): Data matrix (samples x features).
        batch_indices (list or np.ndarray): Group label for each sample.
        BatchNames (dict, optional): Optional mapping from group value to readable name.

    Returns:
        np.ndarray: Cohen's d values, shape = (num_pairs, num_features).
        list: Pair labels, each as a tuple of group names.
    """
    if not isinstance(Data, np.ndarray) or Data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array (samples x features).")
    if not isinstance(batch_indices, (list, np.ndarray)) or np.ndim(batch_indices) != 1:
        raise ValueError("batch_indices must be a 1D list or numpy array.")
    if Data.shape[0] != len(batch_indices):
        raise ValueError("Number of samples in Data must match length of batch_indices.")

    batch_indices = np.array(batch_indices)
    unique_groups = np.unique(batch_indices)

    if len(unique_groups) < 2:
        raise ValueError("At least two unique groups are required to calculate Cohen's d.")

    if BatchNames is None:
        BatchNames = {g: str(g) for g in unique_groups}

    pairwise_d = []
    pair_labels = []

    for g1, g2 in combinations(unique_groups, 2):
        data1 = Data[batch_indices == g1,:]
        data2 = Data[batch_indices == g2,:]

        mean1 = np.mean(data1, axis=0)
        mean2 = np.mean(data2, axis=0)
        std1 = np.std(data1, axis=0, ddof=1)
        std2 = np.std(data2, axis=0, ddof=1)

        pooled_std = np.sqrt((std1 ** 2 + std2 ** 2) / 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            d = (mean1 - mean2) / pooled_std
            d[np.isnan(d)] = 0  # Replace NaNs due to division by zero

        pairwise_d.append(d)
        pair_labels.append((f'{BatchNames[g1]} - {BatchNames[g2]}'))

    return np.array(pairwise_d), pair_labels

# PcaCorr performs PCA on data and computes Pearson correlation of the top N principal components with a batch variable.
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

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



def MahalanobisDistance(Data=None, batch=None, Cov=None,covariates=None):
    """Calculate Mahalanobis distance between pairs of batches in the data
    Parameters:
        Data (np.ndarray): Data matrix where rows are samples and columns are features.
        
        batch (np.ndarray): Batch labels for each sample.

        Cov (np.ndarray, optional): Covariance matrix. If None, it will be computed from Data.
    """

    unique_batches = np.unique(batch)
    if len(unique_batches) < 2:
        raise ValueError("At least two unique batches are required to compute Mahalanobis distance.")
    
    batch_means = {}
    batch_data = {}

    if covariates is not None:
        from numpy.linalg import inv, pinv
        # If covariates are provided, adjust the data accordingly
        """Regress each feature in data on the covariates and return residuals."""
        # Add intercept
        X = np.column_stack([np.ones(covariates.shape[0]), covariates])  # (N, C+1)
        beta = pinv(X) @ Data  # (C+1, X)
        predicted = X @ beta   # (N, X)
        Data = Data - predicted

        return Data

    # Calculate means for each feature in each batch
    for b in unique_batches:
        batch_data[b] = Data[batch == b]
        batch_means[b] = np.mean(batch_data[b], axis=0)

    if Cov is None:
        # Compute pooled covariance matrix from the entire dataset
        Cov = np.cov(Data, rowvar=False)

    mahalanobis_distance = {}

    # Calculate Mahalanobis distance for each pair of batches
    for i,b1 in enumerate(unique_batches):
        for b2 in unique_batches[i+1:]:
            diff = batch_means[b1] - batch_means[b2]
            inv_Cov = np.linalg.inv(Cov)
            distance = np.sqrt(np.dot(np.dot(diff, inv_Cov), diff.T))
            mahalanobis_distance[(b1, b2)] = distance
    return mahalanobis_distance

def mixed_effect_interactions(data,batch,covariates,variable_names):

    """
    Make mixed effect model including cross terms with batch and covariates,

    Parameters: 
        - Data: subjects x features (np.ndarray)
        - batch: subjects x 1 (np.ndarray), batch labels
        - covariates:  subjects x covariates (np.ndarray)
        - variable_names: covariates (list)
    Returns:
        - 
    Raises:
    - ValueError: if Data is not a 2D array or batch is not a
    1D array, or if the number of samples in Data and batch do not match.
    - ValueError: if covariates is not None and not a 2D array
    - ValueError: if variable_names is not None and does not match the number of variables
    """

    if not isinstance(Data, np.ndarray) or Data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array (samples x features).")
    if not isinstance(group_indices, (list, np.ndarray)) or np.ndim(group_indices) != 1:
        raise ValueError("group_indices must be a 1D list or numpy array.")
  
    
    """




# ------------------ CLI Help Only Setup ------------------

# Help functions are set up to provide descriptions of the available functions without executing them.

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