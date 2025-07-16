import numpy as np
from DiagnoseHarmonization import DiagnosticFunctions


def test_pca_corr():
    # Create a sample dataset
    np.random.seed(0)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    # Create a binary indicator for batch
    batch = np.random.randint(0, 2, size=100)  # Binary batch variable

    # Call the PCA correlation function
    explained_variance, score, batchPCcorr = DiagnosticFunctions.PcaCorr(X, batch)

    # Check the shape of the results
    assert score.shape == (100, 4)  # Score should have the same number of samples as X
    assert len(explained_variance) == 4  # Explained variance should match number of features

    # Validate that each variable has 4 correlation values (one per PC)
    for var_stats in batchPCcorr.values():
        assert len(var_stats['correlation']) == 4
        assert len(var_stats['p_value']) == 4


def test_pca_corr_with_covariates():
    np.random.seed(0)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    batch = np.random.randint(0, 2, size=100)  # Binary batch variable
    # Check if the function works with covariates
    covariates = np.random.rand(100, 2)  # Two covariates
    explained_variance_with_cov, score_with_cov, batchPCcorr_with_cov = DiagnosticFunctions.PcaCorr(X, batch, covariates=covariates)
    assert score_with_cov.shape == (100, 4)  # Score should still have the same number of samples
    assert len(explained_variance_with_cov) == 4  # Explained variance should still match number of features

    # Validate that each variable has 4 correlation values (one per PC)
    for var_stats in batchPCcorr_with_cov.values():
        assert len(var_stats['correlation']) == 4
        assert len(var_stats['p_value']) == 4

def test_pca_corr_with_variable_names():
    covariates = np.random.rand(100, 2)  # Two covariates
    np.random.seed(0)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    batch = np.random.randint(0, 2, size=100)  # Binary batch variable
    # Check if the function works with variable names
    variable_names = ['batch', 'covariate1', 'covariate2']
    explained_variance_with_names, score_with_names, batchPCcorr_with_names = DiagnosticFunctions.PcaCorr(X, batch, covariates=covariates, variable_names=variable_names)
    
    assert score_with_names.shape == (100, 4)  # Score should still have the same number of samples
    assert len(explained_variance_with_names) == 4  # Explained variance should still match number of features

    # Validate that each variable has 4 correlation values (one per PC)
    for var_stats in batchPCcorr_with_names.values():
        assert len(var_stats['correlation']) == 4
        assert len(var_stats['p_value']) == 4
"""---------------------------------- Test plotting functions for PCA correlation results ----------------------------------"""

from DiagnoseHarmonization import PlotDiagnosticResults
# Test the PCA plot function with only batch
def test_PC_plot():
    # Create a sample dataset
    np.random.seed(0)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    batch = np.random.randint(0, 2, size=100)  # Binary batch variable
    # Call the PCA correlation function
    explained_variance, score, batchPCcorr = DiagnosticFunctions.PcaCorr(X, batch)
   
    PlotDiagnosticResults.Plot_PC_corr(score, batch)

#%%
# Test the PCA plot function with batch and covariates
def test_PC_plot_with_covariates():
    # Create a sample dataset
    np.random.seed(0)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    batch = np.random.randint(0, 2, size=100)  # Binary batch variable
    covariate_1 = np.random.randint(0,4, size=100)  # One categorical covariate with 4 categories
    covariate_2 = np.random.rand(100)  # One continuous covariate
    covariates =  np.column_stack((covariate_1, covariate_2))  # Combine into a 2D array
    explained_variance_with_cov, score_with_cov, batchPCcorr_with_cov = DiagnosticFunctions.PcaCorr(X, batch, covariates=covariates)

    # Call the PCA correlation function
    PlotDiagnosticResults.Plot_PC_corr(score_with_cov, batch, covariates=covariates)

# Test the PCA plot function with variable names and correlation results
def test_PC_plot_with_variable_names():
    # Create a sample dataset
    np.random.seed(0)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    batch = np.random.randint(0, 2, size=100)  # Binary batch variable
    covariate_1 = np.random.randint(0,4, size=100)  # One categorical covariate with 4 categories
    covariate_2 = np.random.rand(100)  # One continuous covariate
    covariates =  np.column_stack((covariate_1, covariate_2))  # Combine into a 2D array
    variable_names = ['batch', 'Disease category', 'Age']
    explained_variance_with_names, score_with_names, batchPCcorr_with_names = DiagnosticFunctions.PcaCorr(X, batch, covariates=covariates, variable_names=variable_names)

    # Call the PCA correlation function
    PlotDiagnosticResults.Plot_PC_corr(score_with_names, batch, covariates=covariates, variable_names=variable_names,PC_correlations=True)

# %%
