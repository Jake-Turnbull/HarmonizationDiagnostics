from DiagnoseHarmonization import DiagnosticFunctions

import numpy as np

group1 = np.array([1,2,3,4,5])
group2 = np.array([2,3,4,5,6])

def test_cohens_d():
    group = np.random.rand(10,100)
    batch = np.array([0,0,0,0,0,1,1,1,1,1])
    a,b = DiagnosticFunctions.Cohens_D(group, batch)

    assert type(a) == np.ndarray
    assert type(b) == list

def test_pca_corr():
    # Create a sample dataset
    np.random.seed(0)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    batch = np.zeros(100)
    batch[20:40] = batch[20:40] + 1
    batch[40:80] = batch[40:80] + 2
    batch[80:99] = batch[80:99] + 4

    # Call the PCA correlation function
    explained_variance, score, batchPCcorr, pca_ = DiagnosticFunctions.PC_Correlations(X, batch)

    # Check the shape of the results
    assert score.shape == (100, 4)  # Score should have the same number of samples as X
    assert len(explained_variance) == 4  # Explained variance should match number of features

    # Validate that each variable has 3 correlation values (one per PC)
    for var_stats in batchPCcorr.values():
        assert len(var_stats['correlation']) == 4
        assert len(var_stats['p_value']) == 4


        