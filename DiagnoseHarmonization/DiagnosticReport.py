# Diagnostic report generation using DiagnosticFunctions 


def DiagnosticReport(data, batch,
                    covariates=None,
                      batch_names=None,
                        covariate_names=None,
                          save_dir=None,
                            SaveArtifacts=False,
                              rep=None,
                                show=False):
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
        - 'covariate_names': covariate names (list), optional
    Or as individual numpy arrays:
        - data: subjects x features (np.ndarray)
        - batch: batch labels (np.ndarray)
        - covariates: subjects x covariates (np.ndarray), optional
        - covariate_names: covariate names (list), optional

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
    from DiagnoseHarmonization.LoggingTool import StatsReporter
    from DiagnoseHarmonization.LoggingTool import set_report_path

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
        report_path = set_report_path(report, save_dir, report_name="DiagnosticReport.html")
        logger.info(f"Report will be saved to: {report_path}")
        report.log_text(
            f"Analysis started\n"
            f"Number of subjects: {data.shape[0]}\n"
            f"Number of features: {data.shape[1]}\n"
            f"Unique batch: {set(batch)}\n"
            f"HTML report: {report.report_path}\n"
            )
        
        # Check that the data is in the correct format and print the output in the log
        # Some functions can only take Batch as a numeric array, convert now to make a seperate array batch_numeric for these functions and 
        # Create new array for the batch names: batch_names from unique values in batch
        logger.info("Checking data format")

        if isinstance(batch, (list, np.ndarray)):
            batch = np.array(batch)
            if batch.dtype.kind in {'U', 'S', 'O'}:  # string or object (categorical)
                logger.info(f"Original batch categories: {list(set(batch))}")
                logger.info("Creating numeric codes for batch categories")
                batch_numeric, unique = pd.factorize(batch)
                # Return the batch numeric array and the unique batch name matching the numeric code
                logger.info(f"Batch categories: {list(unique)}")
        else:
            raise ValueError("Batch must be a list or numpy array")
        

        # Create batch names from unique values in batch if not provided
        if batch_names is None:
            batch_names = list(set(batch))
        # Check that the length of batch names matches the number of unique batches
        if len(batch_names) != len(set(batch)):
            logger.warning("Length of batch names does not match number of unique batches. Using default names.")
            batch_names = [f"Batch {i+1}" for i in range(len(set(batch)))]
        logger.info(f"Using batch names: {batch_names}")


        
        # Begin the report 
        logger.info("Beginning diagnostic tests")


        # Additive tests first
        logger.info(" The order of tests is as follows: Additive tests, Multiplicative tests, Both")
        logger.info("Additive tests:")
        # Cohen's D test for mean differences
        logger.info("Cohen's D test for mean differences")
        cohens_d_results, pairlabels = DiagnosticFunctions.Cohens_D(data, batch)
        report.log_text("Cohen's D test for mean differences completed")
        PlotDiagnosticResults.Cohens_D_plot(cohens_d_results,pair_labels=pairlabels,rep=report)
        # Add a summary to the results of the Cohen's D test in the log
        logger.info("Cohen's D results summary:")
        # Create summary of number of features with small, medium, large effect sizes
        small_effect = (np.abs(cohens_d_results) < 0.2).sum()
        # Bug fix to make sure that if small effect is zero that it does not save as an array


        medium_effect = ((np.abs(cohens_d_results) >= 0.2) & (np.abs(cohens_d_results) < 0.5)).sum()
        # Bug fix to make sure that if medium effect is zero that it does not save as an array


        large_effect = (np.abs(cohens_d_results) >= 0.6).sum()
        # Bug fix to make sure that if large effect is zero that it does not save as an array


        logger.info(f"Number of features with small effect size (|d| < 0.2): {small_effect}")
        logger.info(f"Number of features with medium effect size (0.2 <= |d| < 0.5): {medium_effect}")
        logger.info(f"Number of features with large effect size (|d| >= 0.5): {large_effect}")
        report.log_text("Cohen's D plot added to report")

        # Mahalanobis distance test for multivariate mean differences
        logger.info("Doing Mahalanobis distance test for multivariate mean differences")
        mahalanobis_results = DiagnosticFunctions.MahalanobisDistance(data, batch,covariates=covariates)
        report.log_text("Mahalanobis distance test for multivariate mean differences completed")
        PlotDiagnosticResults.mahalanobis_distance_plot(mahalanobis_results,rep=report)
        report.log_text("Mahalanobis distance plot added to report")
        # Summary of the Mahalanobis heatmap in the log
        logger.info("Mahalanobis distance results summary:")
        # Create summary of pairwise distances
        pairwise_distances = mahalanobis_results['pairwise_raw']
        for (b1, b2), dist in pairwise_distances.items():
            logger.info(f"Mahalanobis distance between batch {b1} and batch {b2}: {dist:.4f}")
        # Return summary of centroid distances
        centroid_distances = mahalanobis_results['centroid_raw']
        for b, dist in centroid_distances.items():
            logger.info(f"Mahalanobis distance of batch {b} to overall centroid: {dist:.4f}")
        # End of additive tests 


        # Multiplicative tests 
        logger.info("Multiplicative tests:")
        # Levene's test for variance differences
        logger.info("Levene's test for variance differences")
        levene_results = DiagnosticFunctions.Levene_test(data, batch, centre='median')
        report.log_text("Levene's test for variance differences completed")
        # Commenting out the plot for Levene's test as it is not yet implemented
        #PlotDiagnosticResults.plot_Levene(levene_results,report=report)
        #report.log_text("Levene's test plot added to report")
 
        # Variance ratio test between each unique batch pair
        logger.info("Variance ratio test between each unique batch pair")
        variance_ratio = DiagnosticFunctions.Variance_ratios(data, batch, covariates=covariates)
        report.log_text("Variance ratio test between each unique batch pair completed")
        labels = [f"Batch {b1} vs Batch {b2}" for (b1,b2) in variance_ratio.keys()]
        ratio_array = np.array(list(variance_ratio.values()))
        PlotDiagnosticResults.variance_ratio_plot(ratio_array,labels,rep=report)
        report.log_text("Variance ratio test plot added to report")
        # Both additive and multiplicative tests
        logger.info("Both additive and multiplicative tests:")
        logger.info("Generating PCA plots and KS test")


        explained_variance, score, batchPCcorr = DiagnosticFunctions.PcaCorr(data, batch, covariates=covariates,variable_names=covariate_names)

        if covariates is not None:
            logger.info("Covariates provided, checking variable names")
            if covariate_names is None or len(covariate_names) != covariates.shape[1] + 1:
                logger.warning("Variable names not provided or do not match number of covariates + batch. Using default names.")
                covariate_names = ['batch'] + [f'covariate_{i+1}' for i in range(covariates.shape[1])]
            else:
                logger.info(f"Using provided variable names: {covariate_names}")
        else:
            logger.info("No covariates provided")
            covariate_names = ['batch']

        PlotDiagnosticResults.PC_corr_plot(score, batch_numeric, covariates=covariates, variable_names=covariate_names,PC_correlations=True,rep=report,show=False)

        report.log_text("PCA correlation plot added to report")

        # Two-sample Kolmogorov-Smirnov test for distribution differences between each unique batch pair
        logger.info("Two-sample Kolmogorov-Smirnov test for distribution differences between each unique batch pair")
        ks_results = DiagnosticFunctions.KS_test(data, batch, feature_names=None)
        report.log_text("Two-sample Kolmogorov-Smirnov test for distribution differences between each unique batch pair completed")
        PlotDiagnosticResults.KS_plot(ks_results,rep=report)
        report.log_text("Two-sample Kolmogorov-Smirnov test plot added to report")
        # Finalize the report
        logger.info("Diagnostic tests completed")
        
     
def DiagnosticReportLongitudinal():
    # Place holder for future implementation
    return None