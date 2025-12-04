# Test script for testing the power analysis functions in DiagnoseHarmonization/PowerAnalysis.py

import numpy as np
from DiagnoseHarmonization.PowerAnalysis import Cohens_D_PowerAnalysis
import pytest
from DiagnoseHarmonization.DiagnosticFunctions import Cohens_D
import matplotlib.pyplot as plt

def test_Cohens_D_PowerAnalysis():
    # Generate a simple dummy dataset (normal distrubution) with two batches and arbitary shift on one batch (large effect size)

    np.random.seed(42)
    n_samples_per_batch = 100
    n_features = 50
    batch_labels = np.array(['Batch1'] * n_samples_per_batch + ['Batch2'] * n_samples_per_batch)
    # Draw from normal same normal distribution then add arbitary shift to second batch
    data = np.random.normal(loc=0.0, scale=1.0, size=(n_samples_per_batch * 2, n_features))
    data[n_samples_per_batch:, :] += 1.0  # Add shift to second (i.e first N samples_per_batch are Batch1, next N are Batch2)

    # Run Cohen's d and use outputs for power analysis
    cohens_d, pair_labels = Cohens_D(data, batch_labels)

    power_results, figs = Cohens_D_PowerAnalysis(data, batch_labels, cohens_d, pair_labels)
    # Check that power results contain expected keys and values
    assert isinstance(power_results, dict)
    for batch_pair in power_results.keys():
        assert batch_pair in [('Batch1', 'Batch2')]
        for effect_size in [0.1, 0.5, 0.8, 1.2]:
            assert effect_size in power_results[batch_pair]
            for alpha_level in [0.05, 0.01, 0.001]:
                assert alpha_level in power_results[batch_pair][effect_size]
                power_value = power_results[batch_pair][effect_size][alpha_level]
                assert 0.0 <= power_value <= 1.0  # Power should be between 0 and 1 
    # Display the generated figures for visual inspection (optional)
    for title, fig in figs:
        fig.suptitle(title)
        plt.show()

def test_Variance_Ratio_PowerAnalysis():
    # Take two different normal distributions with different variances
    np.random.seed(42)
    n_samples_per_batch = 100
    n_features = 50
    batch_labels = np.array(['Batch1'] * n_samples_per_batch + ['Batch2'] * n_samples_per_batch)
    data = np.random.normal(loc=0.0, scale=1.0, size=(n_samples_per_batch, n_features))
    data2 = np.random.normal(loc=0.0, scale=2.0, size=(n_samples_per_batch, n_features))
    data = np.vstack([data, data2])  # Stack to create full dataset
    # Calculate variance ratios
    from DiagnoseHarmonization.DiagnosticFunctions import Variance_Ratios
    ratio_of_variance = Variance_Ratios(data, batch_labels)
    unique_batches = np.unique(batch_labels)
    from DiagnoseHarmonization.PowerAnalysis import Variance_Ratio_PowerAnalysis
    power_results, figs = Variance_Ratio_PowerAnalysis(data, batch_labels, ratio_of_variance, unique_batches)
    # Check that power results contain expected keys and values
    assert isinstance(power_results, dict)
    for batch_pair in power_results.keys():
        assert batch_pair in [('Batch1', 'Batch2')]
        for effect_size in [0.1, 0.5, 0.8, 1.2]:
            assert effect_size in power_results[batch_pair]
            for alpha_level in [0.05, 0.01, 0.001]:
                assert alpha_level in power_results[batch_pair][effect_size]
                power_value = power_results[batch_pair][effect_size][alpha_level]
                assert 0.0 <= power_value <= 1.0  # Power should be between 0 and 1
    # Display the generated figures for visual inspection (optional)
    for title, fig in figs:
        fig.suptitle(title)
        plt.show()


