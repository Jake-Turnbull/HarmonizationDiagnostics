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

    Args:
        Data:
        Batch:
        Covariate: array of values, each column is one covariate
        Batch_names: N/A needs fixing
        Covariate_names (List): Names of the covariates in the same order as covariate matrix columns
        Save_dir (String): File path to the directory in which to save report and images
        SaveArtifacts (Logical): Save plots as PNG images in Save_dir or current directory
        rep: StatsReporter object (defined in LoggingTool.py) to log outputs to Save_dir
        show (Logical): Show plots as they are generated, default is False


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
        - If SaveArtefacts is set to true, all plots are also saved in the same directory as the report
        - If show is set to true, all plots are also displayed as seperare matplotlib figure windows 
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
        # Define variable to work as linebreak, to be improved at a later date
        line_break_in_text = "-----------------------------------------------------------------------------------------------------------------------------"

        report.text_simple('Summary of dataset:')  
        report.text_simple(line_break_in_text)   
        report.log_text(
            f"Analysis started\n"
            f"Number of subjects: {data.shape[0]}\n"
            f"Number of features: {data.shape[1]}\n"
            f"Unique batches: {set(batch)}\n"
            f"Unique Covariates: {set(covariate_names)}\n"
            f"HTML report: {report.report_path}\n"
            )
        report.text_simple(line_break_in_text)  
        
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
                logger.info(f"Numeric batch codes: {list(set(batch_numeric))}")
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

        # Return the number of samples per batch in the log
        unique_batches, counts = np.unique(batch, return_counts=True)
        report.text_simple("Number of samples per batch:")
        for b, c in zip(unique_batches, counts):
            report.text_simple(f"Batch {b}: {c} samples")
        report.text_simple(line_break_in_text)

        # PLACE HOLDER SPACE FOR CHECKING BATCH SIZES AND ADVSISING MINIMUM SAMPLES PER BATCH FOR RELIABLE TESTS
        # CHECK LIT FOR MIN SAMPLES FOR RELIABLE COHENS D, MAHALANOBIS, LEVENE'S TEST, VARIANCE RATIO TEST, KS TEST
        # ADVISE IN REPORT IF ANY BATCH HAS LESS THAN MINIMUM SAMPLES, CAUTION WHEN RESULTS MAY NOT BE RELIABLE
        # PROVIDE GUIDANCE ON CORRECT APPROACH TO PCA (EG: depening on sample size and well defined effects)
        ##############################################################################################

        ##############################################################################################

        # Begin the reporting of diagnostic tests
        logger.info("Beginning diagnostic tests")


        # Additive tests first
        report.text_simple(" The order of tests is as follows: Additive tests, Multiplicative tests, Tests of distribution")
        report.text_simple(line_break_in_text)   

        logger.info("Additive tests:")

        # Cohen's D test for mean differences
        logger.info("Cohen's D test for mean differences")
        cohens_d_results, pairlabels = DiagnosticFunctions.Cohens_D(data, batch)
        report.log_text("Cohen's D test for mean differences completed")
        PlotDiagnosticResults.Cohens_D_plot(cohens_d_results,pair_labels=pairlabels,rep=report)
        # Add a summary to the results of the Cohen's D test in the log
        # Report the number of features with small, medium and large effect sizes based on Cohen's D thresholds
        # Do for each pairwise batch comparison:
        for i, (b1, b2) in enumerate(pairlabels):
            report.text_simple(f"Summary of Cohen's D results for batch comparison: {b1} vs {b2}")
            cohens_d_pair = cohens_d_results[:, i]
            small_effect = (np.abs(cohens_d_pair) < 0.2).sum()
            medium_effect = ((np.abs(cohens_d_pair) >= 0.2) & (np.abs(cohens_d_pair) < 0.5)).sum()
            large_effect = (np.abs(cohens_d_pair) >= 0.5).sum()
            report.text_simple(
                f"Number of features with small effect size (|d| < 0.2): {small_effect}\n"
                f"Number of features with medium effect size (0.2 <= |d| < 0.6): {medium_effect}\n"
                f"Number of features with large effect size (|d| >= 0.6): {large_effect}\n"
            )
        report.log_text("Cohen's D plot added to report")

        report.text_simple(line_break_in_text)   
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
        logger.info("Pairwise test results")
        for (b1, b2), dist in pairwise_distances.items():
            report.text_simple(f"Mahalanobis distance between batch {b1} and batch {b2}: {dist:.4f}")
        # Return summary of centroid distances
        logger.info("Unique batch to global centroied distance test results") 
        centroid_distances = mahalanobis_results['centroid_raw']
        for b, dist in centroid_distances.items():
            report.text_simple(f"Mahalanobis distance of batch {b} to overall centroid: {dist:.4f}")

        centroid_resid_distance = mahalanobis_results['centroid_resid']
        for b, dist in centroid_resid_distance.items():
            report.text_simple(f"Mahalanobis distance of batch {b} to overall centroid after residualising by covariates: {dist:.4f}")

        # End of additive tests 
        report.text_simple(line_break_in_text)   

        # Multiplicative tests 
        logger.info("Multiplicative tests:")
        # Levene's test for variance differences
        logger.info("Levene's test for variance differences")
        levene_results = DiagnosticFunctions.Levene_test(data, batch, centre='median')
        report.log_text("Levene's test for variance differences completed")
        # Commenting out the plot for Levene's test as it is not yet implemented
        #PlotDiagnosticResults.plot_Levene(levene_results,report=report)
        #report.log_text("Levene's test plot added to report")
        report.text_simple(line_break_in_text)   
        # Variance ratio test between each unique batch pair
        logger.info("Variance ratio test between each unique batch pair")
        variance_ratio = DiagnosticFunctions.Variance_ratios(data, batch, covariates=covariates)
        report.log_text("Variance ratio test between each unique batch pair completed")
        labels = [f"Batch {b1} vs Batch {b2}" for (b1,b2) in variance_ratio.keys()]
        ratio_array = np.array(list(variance_ratio.values()))

        # Summarize variance ratios robustly
        summary_rows = []
        for (b1, b2), ratios in variance_ratio.items():
            ratios = np.array(ratios)
            log_ratios = np.log(ratios)

            # Core summary stats
            mean_log = np.mean(log_ratios)
            median_log = np.median(log_ratios)
            iqr_log = np.percentile(log_ratios, [25, 75])
            prop_higher = np.mean(log_ratios > 0)

            # Back-transform to ratio scale for interpretability
            median_ratio = np.exp(median_log)
            mean_ratio = np.exp(mean_log)

            summary_rows.append({
                "Batch 1": b1,
                "Batch 2": b2,
                "Median log ratio": median_log,
                "Mean log ratio": mean_log,
                "IQR lower": iqr_log[0],
                "IQR upper": iqr_log[1],
                "Prop > 0": prop_higher,
                "Median ratio (exp)": median_ratio,
                "Mean ratio (exp)": mean_ratio,
            })

            # Log the text summary of variance ratio between batches, showing the IQR 
            # and proportion of features with higher variance in batch 1 so that not just mean is used
            logger.info(
                f"Variance ratio {b1} vs {b2}: median log={median_log:.3f} "
                f"(IQR {iqr_log[0]:.3f}–{iqr_log[1]:.3f}), "
                f"{prop_higher*100:.1f}% of features higher in batch {b1}."
            )

        # Make a summary DataFrame for easy export or inclusion in reports
        summary_df = pd.DataFrame(summary_rows)

        # Example: write to your report
        PlotDiagnosticResults.variance_ratio_plot(ratio_array,labels,rep=report)
        report.log_text("Variance ratio plot(s) added to report")


        report.text_simple(line_break_in_text)   

        # Both additive and multiplicative tests
        logger.info("Running PCA")
        explained_variance, score, batchPCcorr = DiagnosticFunctions.PcaCorr(data, batch, covariates=covariates,variable_names=covariate_names)

        if covariates is not None:
            logger.info("Covariates provided, checking variable names")
            if covariate_names is None or len(covariate_names) != covariates.shape[1]:
                logger.warning("Variable names not provided or do not match number of covariates + batch. Using default names.")
                covariate_names = ['batch'] + [f'covariate_{i+1}' for i in range(covariates.shape[1])]
            else:
                logger.info(f"Using provided variable names: {covariate_names}")
        else:
            logger.info("No covariates provided")
            covariate_names = ['batch']

        report.text_simple("Returning correlations of covariates and batch with first four PC's")
        report.text_simple("Returning scatter plots of first two PC's, grouped/coloured by:")
        # Report the names of covariates ued in the PCA correlation plots and the PC1 vs PC2 plot
        report.log_text(f"Variable names used in PCA correlation plots and PC1 vs PC2 plot: {covariate_names}")

        PlotDiagnosticResults.PC_corr_plot(score, batch_numeric, covariates=covariates, variable_names=covariate_names,PC_correlations=True,rep=report,show=False)

        report.log_text("PCA correlation plot added to report")

        report.text_simple(line_break_in_text)   

        # Two-sample Kolmogorov-Smirnov test for distribution differences between each unique batch pair
        logger.info("Two-sample Kolmogorov-Smirnov test for distribution differences between each unique batch pair")
        ks_results = DiagnosticFunctions.KS_test(data, batch, feature_names=None)
        report.log_text("Two-sample Kolmogorov-Smirnov test for distribution differences between each unique batch pair completed")
        PlotDiagnosticResults.KS_plot(ks_results,rep=report)
        plt.close
        report.log_text("Two-sample Kolmogorov-Smirnov test plot added to report")
        # Finalize the report
        logger.info("Diagnostic tests completed")
        
        # Before closing the report, give final summary of results as well as report how to interpret based on effect and sample sizes
        # EG: two sample KS test badly defined sub < 50 samples per batch etc.
         

        logger.info(f"Report saved to: {report.report_path}")

     
def DiagnosticReportLongitudinal():
    # Place holder for future implementation
    return None