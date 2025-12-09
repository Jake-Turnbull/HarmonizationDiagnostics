# Harmonisation diagnostics/evaluation

## Function guidelines

### Main function
- `Diagnosticfunctions.py`: This is IDP-specific evaluation of harmonisation meaning the script will run on each IDP separately. Different metrics are calculated (see below).
	- **Relative percentage difference**: This is within-subject metric for longitudinal data to see if within-subject variability is reduced after harmonisation. The ratio is difference in the IDP by average IDP.
	- **Subject order consistency**: This is spearman correlation calculated (as well as permutation testing) between the timepoints to see if subject order is consistent across timepoints before and after harmonisation.
	- **Diagnostic models**: These are linear mixed effect models fit to predict the harmonised data with age, timepoints, batch (1. Both site and scanner as fixed effects, 2. Site as random effect and scanner as fixed effect), included as fixed effects and subjects as random effects. From these models we primarily get following details:
		- **Subject variability ratio (ICC)**= between-subject variance/(between-subject + within-subject variance); we can check this variability before and after harmonisation
		- **Age and time association**: Effect sizes (beta), CIs and p-values. For age the effects should be preserved or improved. For time, since this is test-rest data, effects should be preserved.
	 	- **Pairwise site comparisons**: pairwise p-values across batches to see if batch effects are removed before and after harmonisation.

### Edit config file and run
- `config-template.json`: edit this to run `run_diagnostics_pipeline.py` on your data
- `python run_diagnostics_pipeline.py –config config.json`