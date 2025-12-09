# Harmonisation diagnostics/evaluation

## Function guidelines

### Main function
- `Diagnosticfunctions.py`: This is IDP-specific evaluation of harmonisation meaning the script will run on each IDP separately. Different metrics are calculated (see below).
	- **Relative percentage difference**: This is within-subject metric for longitudinal data to see if within-subject variability is reduced after harmonisation. Relative % diff is calculated for 2 timepoints within each subject. The ratio is difference in the IDP by average IDP.
    - **Coefficient of variation**: This is within-subject metric for longitudinal data to see if within-subject variability is reduced after harmonisation. CoV is calculated for more than 2 timepoints within each subject.
	- **Subject order consistency**: This is spearman correlation calculated (as well as permutation testing) across the timepoints to see if subject order is consistent across timepoints before and after harmonisation.
    - **Multivariate effect size**: This is Mahalanobis distance calcualted for each batch variable from reference distribution (overall mean and covariance across features). Lower value for a batch indicates batch is closer to the reference.
    - **Additive batch effects**: Additive batch effects were quantified for each IDP by fitting mixed-effects models with IDP as the dependent variable, biological covariates as fixed effects and subjects as random effect. A full model including batch was compared with a reduced model excluding batch using a Kenward–Roger F-test, yielding estimates of the significance and magnitude of additive batch effects.
    - **Multiplicative batch effects**: Multiplicative batch effects were assessed for each IDP by fitting linear mixed-effects models with IDP as the dependent variable, biological covariates as fixed effects and subjects as random effect. Residuals from the full model (including batch) were tested for variance heterogeneity across batches using a Fligner–Killeen test.
	- **Diagnostic models**: These are linear mixed effect models fit to predict the harmonised data with age, timepoints, and batch, which are included as fixed effects and subjects as random effects. From these models we primarily get following details:
		- **Subject variability ratio (ICC)**: between-subject variance/(between-subject + within-subject variance); we can check this variability before and after harmonisation
        - **Within/cross-subject variability**: ratio of within and cross subject variability
		- **Biological preservation**: Effect sizes (beta), CIs and p-values for biological variables
	 	- **Pairwise site comparisons**: To additionally assess whether any residual batch effects remain after harmonisation, we fit a mixed-effects model with the harmonised IDP as the response, including biological covariates (age, timepoint), the batch variable (site and scanner) as the fixed effects, and subjects as random effect. We then conducted pairwise site comparisons using linear contrasts on the fixed effect estimates for batch from the model.

### Edit config file and run
- `config-template.json`: edit this to run `run_diagnostics_pipeline.py` on your data
- `python run_diagnostics_pipeline.py –config config.json`
