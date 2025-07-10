import numpy as np
from DiagnoseHarmonization import DiagnosticFunctions


def test_pca_corr():
    # Create a sample dataset
    np.random.seed(0)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    batch = np.random.rand(100)  # Batch variable

    # Call the PCA correlation function
    pearsonr, explained_variance, score, batchPCcorr = DiagnosticFunctions.PcaCorr(X, batch)

    # Check the shape of the results
    assert score.shape == (100, 5)  # Score should have the same number of samples as X
    assert len(explained_variance) == 5  # Explained variance should match number of features
    assert len(batchPCcorr) == 3  # We only compute correlation for the first 3 PCs