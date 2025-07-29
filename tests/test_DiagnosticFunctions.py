from DiagnoseHarmonization import DiagnosticFunctions

import numpy as np

group1 = np.array([1,2,3,4,5])
group2 = np.array([2,3,4,5,6])

def test_cohens_d():
    group = np.random.rand(10,100)
    batch = np.array([0,0,0,0,0,1,1,1,1,1])
    a,b = DiagnosticFunctions.Cohens_D(group, batch)
    assert np.size(a) ==100

def test_pca_corr():
    # Create a sample dataset
    np.random.seed(0)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    batch = np.zeros(100)
    batch[20:40] = batch[20:40] + 1
    batch[40:80] = batch[40:80] + 2
    batch[80:99] = batch[80:99] + 4

    # Call the PCA correlation function
    explained_variance, score, batchPCcorr = DiagnosticFunctions.PcaCorr(X, batch)

    # Check the shape of the results
    assert score.shape == (100, 4)  # Score should have the same number of samples as X
    assert len(explained_variance) == 4  # Explained variance should match number of features

    # Validate that each variable has 3 correlation values (one per PC)
    for var_stats in batchPCcorr.values():
        assert len(var_stats['correlation']) == 4
        assert len(var_stats['p_value']) == 4

def test_mahalanobis_distance():
# Create a sample dataset

    np.random.seed(0)
    Data = np.random.rand(100, 5)  # 100 samples, 5 features
    batch = np.random.randint(0, 3, size=100)  # 3 unique batches

    # Call the Mahalanobis distance function
    distance = DiagnosticFunctions.MahalanobisDistance(Data, batch)
    print(distance)

    # Check the type of the result
    assert isinstance(distance, dict)

    # Check that we have distances for each pair of batches
    unique_batches = np.unique(batch)
    assert len(distance) == len(unique_batches) * (len(unique_batches) - 1) / 2

    # Check that distances are non-negative
    for key in distance:
        assert distance[key] >= 0
        