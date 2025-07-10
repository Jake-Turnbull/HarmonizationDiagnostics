import argparse
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

# ------------------ Diagnostic Functions ------------------
# Cohens D function calculates the effect size between two groups for each feature.
def Cohens_D(Data1, Data2):
    """Calculate Cohen's d for each feature between two groups."""
    n_features = len(Data1)
    d = [0] * n_features
    for f in range(n_features):
        mean1 = Data1[f].mean()
        mean2 = Data2[f].mean()
        std1 = Data1[f].std()
        std2 = Data2[f].std()
        pooled_std = ((std1 ** 2 + std2 ** 2) / 2) ** 0.5
        d[f] = (mean1 - mean2) / pooled_std if pooled_std != 0 else 0
    return d

# PcaCorr performs PCA on data and computes Pearson correlation of the top N principal components with a batch variable.
def PcaCorr(Data, batch, N_components=3):
    """Perform PCA and correlate top PCs with batch labels."""

    pca = PCA(n_components=N_components)
    score = pca.fit_transform(Data)
    explained_variance = pca.explained_variance_ratio_ * 100
    batch = batch.astype(float)
    batchPCcorr = np.array([
        pearsonr(score[:, i], batch)[0]
        for i in range(min(N_components, score.shape[1]))
    ])
    return pearsonr, explained_variance, score, batchPCcorr



# ------------------ CLI Help Only Setup ------------------

# Help functions are set up to provide descriptions of the available functions without executing them.
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

    return parser

if __name__ == '__main__':
    parser = setup_help_only_parser()
    parser.parse_args()