# Diagnostic report generation using DiagnosticFunctions 
import numpy as np
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from DiagnoseHarmonization import DiagnosticFunctions
from DiagnoseHarmonization import PlotDiagnosticResults
from DiagnoseHarmonization.LoggingTool import StatsReporter

def CrossSectionalReport(
    data,
    batch,
    covariates=None,
    covariate_names=None,
    save_data: bool = False,
    save_data_name: str | None = None,
    save_dir: str | os.PathLike | None = None,
    report_name: str | None = None,
    SaveArtifacts: bool = False,
    rep= None,
    power_analysis: bool = False,
    show: bool = False,
    timestamped_reports: bool = True,
):
    """
    Create a diagnostic report for dataset differences across batches.

    Notes:
      - If `rep` is provided, it will be used as-is (we will set report.save_dir/report.report_name
        if you pass save_dir/report_name). If `rep` is None, a new StatsReporter is created and
        used as a context manager (so it will be closed automatically).
    """

    # Check inputs and revert to defaults as needed
    if save_dir is None:
        save_dir = Path.cwd()
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if report_name is None:
        base_name = "DiagnosticReport.html"
    else:
        base_name = report_name if report_name.endswith(".html") else report_name + ".html"

    if timestamped_reports:
        stem, ext = base_name.rsplit(".", 1)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"{stem}_{timestamp_str}.html"

    # Helper to configure a report object
    def _configure_report(report_obj):
        report_obj.save_dir = save_dir
        report_obj.report_name = base_name
        # write an initial report (optional) and log the path
        rp = report_obj.write_report()  # writes to report_obj.report_path
        report_obj.log_text(f"Initialized HTML report at: {rp}")
        print(f"Report will be saved to: {rp}")
        return report_obj

    # If user passed a report object, use it (do not close it here).
    # Otherwise create one and use it as a context manager so it's closed on exit.
    created_local_report = False
    if rep is None:
        created_local_report = True
        report_ctx = StatsReporter(save_artifacts=SaveArtifacts, save_dir=None)
    else:
        report_ctx = rep

    # If we're using our own, enter the context manager
    if created_local_report:
        ctx = report_ctx.__enter__()  # type: ignore
        report = ctx
    else:
        report = report_ctx

    try:
        logger = report.logger

        # configure save dir/name and write initial stub report
        _configure_report(report)

        line_break_in_text = "-" * 125

        # Basic dataset summary
        report.text_simple("Summary of dataset:")
        report.text_simple(line_break_in_text)
        report.log_text(
            f"Analysis started\n"
            f"Number of subjects: {data.shape[0]}\n"
            f"Number of features: {data.shape[1]}\n"
            f"Unique batches: {set(batch)}\n"
            f"Unique Covariates: {set(covariate_names) if covariate_names is not None else set()}\n"
            f"HTML report: {report.report_path}\n"
        )
        report.text_simple(line_break_in_text)

        # Ensure batch is numeric array where needed
        logger.info("Checking data format")
        if isinstance(batch, (list, np.ndarray)):
            batch = np.array(batch)
            if batch.dtype.kind in {"U", "S", "O"}:  # string/object categorical
                logger.info(f"Original batch categories: {list(set(batch))}")
                logger.info("Creating numeric codes for batch categories")
                batch_numeric, unique = pd.factorize(batch)
                logger.info(f"Numeric batch codes: {list(set(batch_numeric))}")
                # keep string labels in `batch` if plotting expects them; numeric conversions can be used inside tests as needed
        else:
            raise ValueError("Batch must be a list or numpy array")

        # Prepare save-data dict if requested
        if save_data:
            data_dict = {}
            data_dict["batch"] = batch
            if covariates is not None:
                for i in range(covariates.shape[1]):
                    if covariate_names is not None and i < len(covariate_names):
                        cov_name = covariate_names[i]
                    else:
                        cov_name = f"covariate_{i+1}"
                    data_dict[cov_name] = covariates[:, i]
            if save_data_name is None:
                save_data_name = "DiagnosticReport_InputData.csv"
        else:
            data_dict = None

        # Samples per batch
        unique_batches, counts = np.unique(batch, return_counts=True)
        report.text_simple("Number of samples per batch:")
        for b, c in zip(unique_batches, counts):
            report.text_simple(f"Batch {b}: {c} samples")
        report.text_simple(line_break_in_text)

        # Begin tests
        logger.info("Beginning diagnostic tests")
        report.text_simple(" The order of tests is as follows: Additive tests, Multiplicative tests, Tests of distribution")
        report.text_simple(line_break_in_text)

        report.log_section("Z-score visualization", "Z-score normalization visualization")

        logger.info("Generating Z-score normalization visualization")
        report.text_simple("Z-score normalization (median-centred) visualization across batches, " \
        "the further the histograms are apart, the larger the mean batch differences on average across features." \
        "We show also here a heatmap sorted by batch for further visualisation of batch effects, " \
        "larger blocks of similar colours that are far from zero indicate larger batch effects compared to average across batches ")
        zscored_data = DiagnosticFunctions.z_score(data)
        PlotDiagnosticResults.Z_Score_Plot(zscored_data, batch, rep=report)
        report.log_text("Z-score normalization visualization added to report")
        report.text_simple(line_break_in_text)
        # ---------------------
        # Additive tests
        # ---------------------
        report.log_section("cohens_d", "Cohen's D test for mean differences")
        logger.info("Cohen's D test for mean differences")
        cohens_d_results, pairlabels = DiagnosticFunctions.Cohens_D(data, batch)
        report.text_simple("Cohen's D test for mean differences completed")

        # Plot (PlotDiagnosticResults should call rep.log_plot internally; our report.log_section ensures plots are attached)
        PlotDiagnosticResults.Cohens_D_plot(cohens_d_results, pair_labels=pairlabels, rep=report)
        report.log_text("Cohen's D plot added to report")

        # Summaries per pair
        for i, (b1, b2) in enumerate(pairlabels):
            report.text_simple(f"Summary of Cohen's D results for batch comparison: {b1} vs {b2}")
            cohens_d_pair = cohens_d_results[i, :]
            if save_data:
                data_dict[f"CohensD_{b1}_vs_{b2}"] = cohens_d_pair
            small_effect = (np.abs(cohens_d_pair) < 0.2).sum()
            medium_effect = ((np.abs(cohens_d_pair) >= 0.2) & (np.abs(cohens_d_pair) < 0.5)).sum()
            large_effect = (np.abs(cohens_d_pair) >= 0.5).sum()
            report.text_simple(
                f"Number of features with small effect size (|d| < 0.2): {small_effect}\n"
                f"Number of features with medium effect size (0.2 <= |d| < 0.6): {medium_effect}\n"
                f"Number of features with large effect size (|d| >= 0.6): {large_effect}\n"
            )

        report.text_simple(line_break_in_text)

        # Mahalanobis
        report.log_section("mahalanobis", "Mahalanobis distance test")
        logger.info("Doing Mahalanobis distance test for multivariate mean differences")
        mahalanobis_results = DiagnosticFunctions.Mahalanobis_Distance(data, batch, covariates=covariates)
        report.log_text("Mahalanobis distance test for multivariate mean differences completed")
        PlotDiagnosticResults.mahalanobis_distance_plot(mahalanobis_results, rep=report)
        report.log_text("Mahalanobis distance plot added to report")

        # Summaries from mahalanobis_results
        pairwise_distances = mahalanobis_results.get("pairwise_raw", {})
        for (b1, b2), dist in pairwise_distances.items():
            report.text_simple(f"Mahalanobis distance between {b1} and {b2}: {dist:.4f}")

        centroid_distances = mahalanobis_results.get("centroid_raw", {})
        for b, dist in centroid_distances.items():
            report.text_simple(f"Mahalanobis distance of {b} to overall centroid: {dist:.4f}")

        centroid_resid_distance = mahalanobis_results.get("centroid_resid", {})
        for b, dist in centroid_resid_distance.items():
            report.text_simple(f"Mahalanobis distance of {b} to overall centroid after residualising by covariates: {dist:.4f}")

        if save_data:
            for b, dist in centroid_distances.items():
                data_dict[f"Mahonalobis_Centroid_Batch{b}"] = dist
            for b, dist in centroid_resid_distance.items():
                data_dict[f"Mahonalobis_Centroid_Resid_Batch{b}"] = dist

        report.text_simple(line_break_in_text)


        # ---------------------
        # Mixed model tests
        # ---------------------
        logger.info("Beginning LMM diagnostics")
        report.log_section("lmm_diagnostics", "Linear mixed effects diagnostics (batch + covariates)")
        report.text_simple("Fitting per-feature LMMs (random intercept for batch). Where LMM fails or batch variance is zero we fallback to OLS fixed-effects.")

        # run LMM diagnostics
        lmm_results_df, lmm_summary = DiagnosticFunctions.Run_LMM(data, batch, covariates=covariates,
                                                feature_names=None,
                                                covariate_names=covariate_names,
                                                min_group_n=2)

        report.text_simple("LMM diagnostics completed.")
        report.log_text("LMM results table added to report")

        # add summary text
        report.text_simple(
            f"Number of features analyzed: {lmm_summary.get('n_features', 0)}\n"
            f"Features where LMM succeeded: {lmm_summary.get('succeeded_LMM', 0)}\n"
            f"Features using fallback (OLS or skipped): {lmm_summary.get('used_fallback', 0)}"
        )

        # list common notes
        note_lines = []
        for tag, count in sorted(lmm_summary.items(), key=lambda x: -x[1])[:10]:
            if tag == 'n_features':
                continue
            note_lines.append(f"{tag}: {count}")
        report.text_simple("LMM diagnostics notes (top):\n" + "\n".join(note_lines))

        # Save DF if needed
        if save_data:
            data_dict['LMM_results_df'] = lmm_results_df
            data_dict['LMM_summary'] = lmm_summary
        
        report.text_simple("Histogram of ICC (proportion of variance explained by batch):")
        # How to interpret ICC:
        report.text_simple("Intraclass Correlation Coefficient (ICC) is the ratio of variance due to batch effects to the total variance (batch + residual). \n" 
        "It quantifies the extent to which batch membership explains variability in the data.")
        report.text_simple(
            "Interpretation of ICC values:\n"
            "- ICC close to 0: Little to no variance explained by batch; suggests minimal batch effect.\n"
            "- ICC around 0.1-0.3: Small batch effect; may be acceptable depending on context.\n"
            "- ICC around 0.3-0.5: Moderate batch effect; consider further investigation or correction.\n"
            "- ICC above 0.5: Strong batch effect; likely requires correction to avoid confounding.\n"
        )
        try:
            icc_nonan = lmm_results_df['ICC'].dropna()
            if len(icc_nonan) > 0:
                plt.figure(figsize=(6, 3))
                plt.hist(icc_nonan, bins=30)
                plt.title("Distribution of ICC across features")
                report.log_plot(plt, caption="Histogram of ICC (batch variance proportion)")
                plt.close()
        except Exception:
            logger.exception("Could not produce ICC histogram")

        # Plot ICC per feature, use simple bar plot for now, replace with plotting functions at later date:
        if len(icc_nonan) > 0:
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(icc_nonan)), icc_nonan)
            plt.xlabel("Feature index")
            plt.ylabel("ICC")
            plt.title("ICC values per feature")
            report.log_plot(plt, caption="ICC values per feature")
            plt.close()

        # Plot conditional and marginal R^2 per feature, indicate what each means for interpretation
        report.text_simple("Marginal R² represents the variance explained by fixed effects (covariates)\n"
                           "while Conditional R² represents the variance explained by both fixed and random effects (batch + covariates).")
        lmm_r = lmm_results_df[['R2_marginal', 'R2_conditional']].dropna()
        if len(lmm_r) > 0:
            plt.figure(figsize=(10, 4))
            plt.plot(lmm_r['R2_marginal'].values, label='Marginal R²', alpha=0.7)
            plt.plot(lmm_r['R2_conditional'].values, label='Conditional R²', alpha=0.7)
            plt.xlabel("Feature index")
            plt.ylabel("R² value")
            plt.title("Marginal and Conditional R² values per feature")
            plt.legend()
            report.log_plot(plt, caption="Marginal and Conditional R² values per feature")
            plt.close()

        # ---------------------

        # ---------------------
        # Multiplicative tests
        # ---------------------
        report.log_section("levene", "Levene (Brown-Forsythe) test for variance differences")
        logger.info("Levene's test for variance differences")
        levene_results = DiagnosticFunctions.Levene_Test(data, batch, centre="median")
        report.log_text("Levene's test for variance differences completed")
        # Plot for Levene if implemented in PlotDiagnosticResults
        try:
            PlotDiagnosticResults.plot_Levene(levene_results, report=report)
            report.log_text("Levene test plot added to report")
        except Exception:
            logger.debug("Levene plot not implemented or failed to run", exc_info=True)

        report.text_simple(line_break_in_text)

        # Variance ratio
        report.log_section("variance_ratio", "Variance ratio test (pairwise)")
        logger.info("Variance ratio test between each unique batch pair")
        variance_ratio = DiagnosticFunctions.Variance_Ratios(data, batch, covariates=covariates)
        report.log_text("Variance ratio test between each unique batch pair completed")

        labels = [f"Batch {b1} vs Batch {b2}" for (b1, b2) in variance_ratio.keys()]
        ratio_array = np.array(list(variance_ratio.values()))

        summary_rows = []
        for (b1, b2), ratios in variance_ratio.items():
            ratios = np.array(ratios)
            log_ratios = np.log(ratios)
            mean_log = np.mean(log_ratios)
            median_log = np.median(log_ratios)
            iqr_log = np.percentile(log_ratios, [25, 75])
            prop_higher = np.mean(log_ratios > 0)
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
            if save_data:
                data_dict[f"VarianceRatio_Batch{b1}_vs_Batch{b2}"] = ratios
                data_dict[f"MedianLogVarianceRatio_Batch{b1}_vs_Batch{b2}"] = median_log
                data_dict[f"MeanLogVarianceRatio_Batch{b1}_vs_Batch{b2}"] = mean_log
                data_dict[f"IQRLowerLogVarianceRatio_Batch{b1}_vs_Batch{b2}"] = iqr_log[0]
                data_dict[f"IQRUpperLogVarianceRatio_Batch{b1}_vs_Batch{b2}"] = iqr_log[1]
                data_dict[f"PropHigherLogVarianceRatio_Batch{b1}_vs_Batch{b2}"] = prop_higher
                data_dict[f"MedianVarianceRatioExp_Batch{b1}_vs_Batch{b2}"] = median_ratio
                data_dict[f"MeanVarianceRatioExp_Batch{b1}_vs_Batch{b2}"] = mean_ratio

            logger.info(
                f"Variance ratio {b1} vs {b2}: median log={median_log:.3f} "
                f"(IQR {iqr_log[0]:.3f}–{iqr_log[1]:.3f}), "
                f"{prop_higher*100:.1f}% of features higher in batch {b1}"
            )

        summary_df = pd.DataFrame(summary_rows)
        PlotDiagnosticResults.variance_ratio_plot(ratio_array, labels, rep=report)
        report.log_text("Variance ratio plot(s) added to report")
    
        report.text_simple(line_break_in_text)

        # ---------------------
        # PCA and clustering
        # ---------------------
        report.log_section("pca", "PCA & covariate correlations")
        logger.info("Running PCA")
        if covariates is not None:
            if covariate_names is None or len(covariate_names) != covariates.shape[1]:
                logger.warning("Variable names not provided or do not match number of covariates. Using defaults.")
                covariate_names = ["batch"] + [f"covariate_{i+1}" for i in range(covariates.shape[1])]
            else:
                logger.info(f"Using provided variable names: {covariate_names}")
        else:
            covariate_names = ["batch"]

        variable_names = ["batch"] + covariate_names
        explained_variance, score, batchPCcorr, pca = DiagnosticFunctions.PC_Correlations(
            data, batch, covariates=covariates, variable_names=variable_names
        )

        report.text_simple("Returning correlations of covariates and batch with first four PC's")
        report.text_simple("Returning scatter plots of first two PC's, grouped/coloured by:")
        report.log_text(f"Variable names used in PCA correlation plots and PC1 vs PC2 plot: {covariate_names}")

        PlotDiagnosticResults.PC_corr_plot(
            score, batch, covariates=covariates, variable_names=covariate_names,
            PC_correlations=True, rep=report, show=False
        )
        report.log_text("PCA correlation plot added to report")

        # Clustering diagnostics (if enough samples)
        n_samples = data.shape[0]
        n_clusters = len(np.unique(batch))
        if n_samples >= n_clusters + 1:
            cumulative_variance = np.cumsum(explained_variance)
            n_pcs_for_clustering = np.searchsorted(cumulative_variance, 70) + 1
            if n_pcs_for_clustering < 2:
                n_pcs_for_clustering = 2
            logger.info(f"Number of PCs to explain 70% variance: {n_pcs_for_clustering}")
            n_clusters_for_kmeans = unique_batches.shape[0]
            PlotDiagnosticResults.pc_clustering_diagnostics(
                PrincipleComponents=score,
                batch=batch,
                covariates=covariates,
                variable_names=covariate_names,
                n_pcs_for_clustering=n_pcs_for_clustering,
                n_clusters_for_kmeans=n_clusters_for_kmeans,
                rep=report,
                random_state=0,
                show=False,
            )
            report.log_text("Clustering diagnostics plot added to report")
        else:
            logger.warning("Dataset size insufficient for clustering diagnostics, skipping K-means clustering")
            report.log_text("Clustering diagnostics skipped due to insufficient dataset size")

        report.text_simple(line_break_in_text)

        # ---------------------
        # Distribution tests (KS)
        # ---------------------
        report.log_section("ks", "Two-sample Kolmogorov-Smirnov tests")
        logger.info("Two-sample Kolmogorov-Smirnov test for distribution differences between each unique batch pair")
        ks_results = DiagnosticFunctions.KS_Test(data, batch, feature_names=None)
        report.log_text("Two-sample Kolmogorov-Smirnov test completed")

        for key, value in ks_results.items():
            if key != "params":
                logger.info(f"Key: {key}, Value type: {type(value)}")

        report.text_simple(
            "- each value is a dict with:\n"
            "    'statistic': np.array of D statistics (length n_features)\n"
            "    'p_value': np.array of p-values (nan where test not run)\n"
            "    'p_value_fdr': np.array of BH-corrected p-values (if do_fdr else None)\n"
            "    'n_group1': array of sample counts per feature for group1\n"
            "    'n_group2': array of counts for group2\n"
        )

        if save_data:
            for key, value in ks_results.items():
                if key != "params":
                    data_dict[f"KS_Stat_{key}"] = value["statistic"]
                    data_dict[f"KS_PValue_{key}"] = value["p_value"]
                    if value.get("p_value_fdr") is not None:
                        data_dict[f"KS_PValueFDR_{key}"] = value["p_value_fdr"]

        PlotDiagnosticResults.KS_plot(ks_results, rep=report)
        report.log_text("Two-sample Kolmogorov-Smirnov test plot added to report")

        # Finalize
        logger.info("Diagnostic tests completed")
        logger.info(f"Report saved to: {report.report_path}")

        # Save data dictionary as csv if requested
        if save_data and data_dict is not None:
            import csv
            csv_path = os.path.join(save_dir, save_data_name)
            # If data_dict values are arrays, convert to something writable - here we write a single-row dict with lists/stringified arrays
            serializable = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in data_dict.items()}
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(serializable.keys()))
                writer.writeheader()
                writer.writerow(serializable)
        
        report.log_section("Summary","Summary of Diagnostic Report and Advice")

        # Report the major findings in brief and provide advice on harmonisation method to apply in this context
        # Summarise the biggest additive differences between batches (Cohen's d, Mahalanobis)

        # Check size of each batch
        batch_sizes = {b: np.sum(batch == b) for b in unique_batches}
        min_batch_size = min(batch_sizes.values())
        max_batch_size = max(batch_sizes.values())

        # Check if large differences in batch sizes (if the difference between smallest and largest batch is more than double)
        if max_batch_size > 2 * min_batch_size:
            report.text_simple(
                f"Note: Large differences in batch sizes detected (smallest batch size: {min_batch_size}, largest batch size: {max_batch_size}). "
                "When applying harmonisation, be aware that most methods will apply a greater correction to smaller batches. " \
                "This isn't inherently problematic but may not be exactly what you want so proceed with caution or consider using the larger batch as a reference batch explicitly so that it remains unchanged in harmonisation"
            )
        
        return data_dict if save_data else None

    finally:
        # If we created the local report context, close it properly
        if created_local_report:
            # call __exit__ on the context-managed report (no exception info)
            report_ctx.__exit__(None, None, None)  # type: ignore

