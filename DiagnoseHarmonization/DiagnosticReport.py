# Diagnostic report generation using DiagnosticFunctions 

from DiagnoseHarmonization.LoggingTool import StatsReporter


def DiagnosticReport(data, batch, covariates=None, variable_names=None,save_dir=None, SaveArtifacts=False):
    """
    Create a diagnostic report for dataset differences across batches, taking into account covariates
    when relevant.
    The different tests used are all defined in DiagnosticFunctions.py and the plots in PlotDiagnosticResults.py.
    The following tests are included:

    Additive components:
        - Cohen' D test for mean differences (standardized mean difference)
        - Mahalanobis distance test for multivariate mean differences
    
    Multiplicative components:
        - Levene's test for variance differences (set as Brown-Forsythe test for rubustness)
        - Variance ratio test between each unique batch pair 
    
    Both:
        - PCA visualization of data colored by batch
        - PCA correlation with batch and covariates 
        - Two-sample Kolmogorov-Smirnov test for distribution differences between each unique batch pair 

    Args:
    Arguments can be given as a pandas dataframe or as individual numpy arrays.

    As a pandas dataframe have the following columns:
        - 'data': subjects x features (np.ndarray)
        - 'batch': batch labels (np.ndarray)
        - 'covariates': subjects x covariates (np.ndarray), optional
        - 'variable_names': covariate names (list), optional
    Or as individual numpy arrays:
        - data: subjects x features (np.ndarray)
        - batch: batch labels (np.ndarray)
        - covariates: subjects x covariates (np.ndarray), optional
        - variable_names: covariate names (list), optional

    Returns:
        - report: a HTML file containing the outputs from each diagnostic function (from DiagnosticFunctions.py) and 
        and the corresponding plots (from PlottingFunctions.py)
    Raises:
        - ValueError: if Data is not a 2D array or batch is not a
        1D array, or if the number of samples in Data and batch do not match. 
    """
# Import the necessary libraries and functions from diagnostic functions and plotting functions
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import os
    import matplotlib.pyplot as plt
    from DiagnoseHarmonization import DiagnosticFunctions
    from DiagnoseHarmonization import PlotDiagnosticResults

# Start the log defined using StatsReporter from LoggingTool.py 
    with StatsReporter(save_artifacts=SaveArtifacts, save_dir=None) as report:
        logger = report.logger

        # Run the diagnostic functions and log their outputs
        logger.info("Starting report generation")

        if save_dir is None:
            logger.info("No save directory specified, saving to current working directory")
            save_dir = os.getcwd()
        else:
            logger.info(f"Saving to directory: {save_dir}")
        report_path = Path(save_dir) / "DiagnosticReport.html"
        report.set_report_path(report_path)
        logger.info(f"Report will be saved to: {report.report_path}")
        report.log_text(
            f"Analysis started\n"
            f"Number of subjects: {data.shape[0]}\n"
            f"Number of features: {data.shape[1]}\n"
            f"Unique batch: {set(batch)}\n"
            f"HTML report: {report.report_path}\n"
            )
        # Check that the data is in the correct format and print the output in the log

        if isinstance(batch, (list, np.ndarray)):
            batch = np.array(batch)
            if batch.dtype.kind in {'U', 'S', 'O'}:  # string or object (categorical)
                logger.info("Converting categorical batch to numeric codes")
                batch, unique = pd.factorize(batch)
                logger.info(f"Batch categories: {list(unique)}")
        else:
            raise ValueError("Batch must be a list or numpy array")
        
        # Begin the report 
        logger.info("Beginning diagnostic tests")

        # Additive tests first

        logger.info("Running additive tests")
        """"---------------------------------------- Cohen's d ----------------------------------------"""
        # Cohen's D test for mean differences, returns the pairwise Cohen's D values and the labels for each pair
        logger.info("Running Cohen's D test for mean differences")
        [cohen_d_results,labels] = DiagnosticFunctions.cohens_d_test(data, batch)

        logger.info("Cohen's D test completed")
        # Loop over each unique pair and return the number of features with a Cohen's D above 0.2, 0.5 and 0.8 (small, medium and large effect sizes)
        unique_pairs = []
        for i in range(len(labels)):
            pair = labels[i]
            if pair not in unique_pairs and (pair[1], pair[0]) not in unique_pairs:
                unique_pairs.append(pair)
                d_values = cohen_d_results[i]
                n_small = np.sum(np.abs(d_values) >= 0.2)
                n_medium = np.sum(np.abs(d_values) >= 0.5)
                n_large = np.sum(np.abs(d_values) >= 0.8)
                logger.info(f"Cohen's D between batches {pair[0]} and {pair[1]}: {n_small} features with small effect size (d>=0.2), {n_medium} features with medium effect size (d>=0.5), {n_large} features with large effect size (d>=0.8)")

        # Plot the Cohen's D results, must loop over each unique pair to save the plots
        for i in range(len(labels)):
            pair = labels[i]
            if pair not in unique_pairs and (pair[1], pair[0]) not in unique_pairs:
                unique_pairs.append(pair)
                d_values = cohen_d_results[i]
                PlotDiagnosticResults.Cohens_D(d_values, labels=pair, df=None)
                logger.info(f"Cohen's D plot for batches {pair[0]} and {pair[1]} completed")
                # Save the figures into the HTML report
                report.log_plot(plt.gcf(), caption=f"Cohen's D Effect Size between batches {pair[0]} and {pair[1]}")
                plt.close('all')        
                # Clear the current figure to avoid overlap
        
        """"---------------------------------------- Mahalanobis Distance ----------------------------------------"""
        # Mahalanobis distance test for multivariate mean differences, returns the Mahalanobis distances and the p-values for each batch
        logger.info("Running Mahalanobis distance test for multivariate mean differences")
        [mahalanobis_results, p_values] = DiagnosticFunctions.mahalanobis_distance_test(data, batch)
        logger.info("Mahalanobis distance test completed")
        # Log the Mahalanobis distances and p-values for each batch
        for i in range(len(mahalanobis_results)):
            logger.info(f"Mahalanobis distance for batch {i}: {mahalanobis_results[i]}, p-value: {p_values[i]}")
        # Plot the Mahalanobis distance results
        #PlotDiagnosticResults.Mahalanobis(mahalanobis_results, p_values) # Currenltly not added so commented out
        # Save the figure into the HTML report
        #report.log_plot(plt.gcf(), caption="Mahalanobis Distance between Batches")
        #plt.close()

        logger.info("Mahalanobis distance plot completed")
        
        # Multiplicative/scale tests next
        logger.info("Running multiplicative tests")
        """"---------------------------------------- Levene's Test ----------------------------------------"""
        # Levene's test for variance differences, returns the Levene's test statistics and p-values for each feature
        logger.info("Running Levene's test for variance differences")
        [levene_results, p_values] = DiagnosticFunctions.levenes_test(data, batch)
        logger.info("Levene's test completed")
        # Log the number of features with significant variance differences at p<0.05
        alpha = 0.05
        n_significant = np.sum(p_values < alpha)
        logger.info(f"Levene's test: {n_significant} features with significant variance differences at p<{alpha}")

        # Plot the Levene's test results, currently not implemented so use basic bar plot
        plt.figure(figsize=(10,6))
        plt.hist(p_values, bins=50, color='skyblue', edgecolor='black')
        plt.axvline(x=alpha, color='red', linestyle='--', label=f'p<{alpha}')
        plt.xlabel('p-value')
        plt.ylabel('Frequency')
        plt.title("Levene's Test p-value Distribution")
        plt.legend()
        # Save the figure into the HTML report
        report.log_plot(plt.gcf(), caption="Levene's Test p-value Distribution")
        plt.close('all')
        logger.info("Levene's test plot completed")
        
        """"---------------------------------------- Variance Ratio Test ----------------------------------------"""
    
    