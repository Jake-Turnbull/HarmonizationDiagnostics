# DiagnoseHarmonize version 0.0.1

DiagnoseHarmonize is an in-development library for the streamline application and assesment of harmonization algorithms at the summary measure level, as well as the establishment of a centralised location for popular existing harmonization methods that are well validated within the literature.
We plan to show in an upcoming paper that the systematic evaluation of different components of the batch effect and subsequent reporting is not only beneficial for choosing a good harmonisation strategy, but essential for evaluating how well it has worked.

Load in different components of the module by calling:
    from DiagnoseHarmonization import ModuleName

If you find any issues or bugs in any part of the code, please raise it as an issue or alternatively contact the following:

Jake Turnbull: [jacob.turnbull@ndcn.ox.ac.uk](mailto:jacob.turnbull@ndcn.ox.ac.uk)

Gaurav Bhalerao: [Gaurav.bhalerao@ndcn.ox.ac.uk](Gaurav.bhalerao@ndcn.ox.ac.uk)

## Overview

This library was developed for the streamline analysis and application of harmonisation for MRI data. Consistent reporting of the different components of batch differences should be carried out both pre and post harmonisation both in order to confirm that harmonisation has been successful but also that harmonisation was needed. While this tool was developed for MRI data, there is no reason why it cannot be used in other research scenarios.

The purpose of harmonisation is to remove technical variation that is driven by differences in how the data was acquired (e.g from different sites) and is seperate from any signals of interest. This must be done while preserving important biological signals in the data.

Harmonisation efficacy should therefore be measured in two broad categories, the first is the reduction or removal of a 'batch effect', which is a broad term that describes the difference between datasets that is driven by unwanted technical differences. The second category is through the preservation of meaningful biological signal.

Here we present a set of functions that can be applied to your data in order to assess the severity, nature and distribution of batch effects for multi-batch data across features in order to provide a 'diagnosis' of the batch differences and advise a best harmonisation method to use.

We provide functions to do this for the following data scenarios but generally, harmonisation is goal specific, so its integration into experimental design should be carefully considered, with the diagnostic reports being a viable method of advising your experimental design.

## DiagnosticReport.py

CrossSectionalReport():
    Assumes every row is an independant sample, focuses on statistical differences between batches and the tests for potential confounding of batch with any given covariates in the data. Should be used before applying harmonisation to assess severity of mean difference, scaling differenes, differences in distributions provides visualisation of effects and covariates through PC clustering and correlations.

LongitudinalReport()
    Additional vector of subject ID's required. Longitudinal harmonisation efficacy must have an aditional goal of ensuring that between subject variability is preserved or recovered through harmonisation. Assess additive, multiplicative and distributional component of batch assuming that batch effects all participants meaures the same for each observation (feature). Ensure consitency of subject ranking between sites (i.e if person A has larger ROI volume in sites A, B and C then person B, then person B's measures should be smaller at all sites than just A to A)

UnknownBatchReport()
    New methods have used image quality metrics to estimate technical differences between batches. These are continuous variables so quantifying severity is best done through modelling approaches rather than by using a grouping variable. This script is a Work In Progress and will be added at a later data

## LoggingTool.py

    Enhanced logging and HTML report generation for diagnostic reports.
    Provides the StatsReporter class that allows logging text and plots, organizing them into sections, and writing a structured HTM report with a table of contents.
    If individuals would like to use this library to create their own analysis scripts, we suggest using the logging tool as an easy way to organise and return results (see script for more detail)

    Functions:
    - log_section(section_id, title): mark a new named section in the log 
    - log_plot(fig, caption, section=None): attach a plot to a section (defaults to last section)
    - write_report(...) builds a TOC with hyperlinks and places each section's plots immediately after its logs.

## DiagnosticFunctions.py

    Definitions for each of the functions called by the different reporting tools in DiagnosticReport.py are written here.
    Each function will show either the additive, multiplicative or distribution difference between batches. Additionally, covariate effects using Linear models (either mixed effects or if this cannot be fit, fixed effects through OLS)
    
    Collection of statistical functions to assess and visualise batch effects in tabular data.
    Functions:
    - Cohens_D: Calculate Cohen's d effect size between batches for each feature.
    - Mahalanobis_Distance: Calculate Mahalanobis distance between batches.
    - PC_Correlations: Perform PCA and correlate top PCs with batch and covariates.
    - fit_lmm_safe: Robustly fit a Linear Mixed Model with fallbacks and diagnostics.
    - Variance_Ratios: Calculate variance ratios between batches for each feature.
    - KS_Test: Performs two-sample Kolmogorov-Smirnov test between batches for each feature.

## PlotDiagnosticResults.py

    Complementary plotting functions for the functions in DiagnosticFunctions.py
    Functions:
    - Z_Score_Plot: Plot histogram and heatmap of Z-scored data by batch.
    - Cohens_D_plot: Plot Cohen's d effect sizes with histograms.
    - variance_ratio_plot: Plot variance ratios between batches.
    - PC_corr_plot: Generate PCA diagnostic plots including scatter plots and correlation heatmaps.
    - PC_clustering_plot: K-means clustering and silhouette analysis of PCA results by batch.
    - Ks_Plot: Plot KS statistic between batches.

## HarmonizationFunctions.py

    Collection of widely used functions for applying harmonisation to tabular data:
        HarmonizationFunctions.combat(data, batch, mod, parametric,
                    DeltaCorrection=True, UseEB=True, ReferenceBatch=None,
                    RegressCovariates=False, GammaCorrection=True)          
            Apply combat harmonisation to your data:
            Mandatory: 
                dat: should be M x N (features by observations) array
                Batch: N length array or list 
                mod: N x C array where C is your number of covariates
                parametric: Whether to set parametric to true or false (should be set to True, setting parametric to false results in longer run time and is unstable, may be fixed n future implementation)             
            Optional: 
                DeltaCorrection: Whether to apply scaling correction (default true)
                UseEB: Use empirical Bayes to estimate location scale correctio (default true)
                ReferenceBatch: Set batch label in batch to be the reference, no L/S correction is applied to this batch and this batches mean and variance are used when estimating batch corrections
                RegressCovariates: Should covariate effects estimated from mod be added back into data or data returned as residuals (default False)
                GammaCorrection: Whether to apply mean shift correction (default true)
