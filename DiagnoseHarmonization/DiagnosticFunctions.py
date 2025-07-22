import pandas as pd
import numpy as np
from scipy.stats import zscore  
import statsmodels.formula.api as smf
import warnings
from sklearn.preprocessing import scale as zscore
from itertools import combinations
from scipy.stats import stats  
import statsmodels.formula.api as smf
from scipy import stats

def rpd_fun(x):
    return abs(x[0] - x[1]) / x.mean() * 100

def rpd(x):
    """
    Compute Relative Percent Difference (RPD) between two values.
    """
    if len(x) != 2:
        return float('nan')  # or raise an error
    return abs(x.iloc[0] - x.iloc[1]) / x.mean() * 100

def compute_within_subject_rpd(data, methods=None, start_col=None, subject_col='subjectID'):
    """
    Computes within-subject RPD for each method.

    Parameters:
    - data: DataFrame with repeated measures
    - methods: list of method column names (optional)
    - start_col: integer column index to start selecting methods (optional)
    - subject_col: name of the subject identifier column

    Returns:
    - DataFrame with one row per subject and one column per method's RPD
    """
    if methods is None:
        if start_col is None:
            raise ValueError("Either 'methods' or 'start_col' must be provided.")
        methods = list(data.columns[start_col:])

    rpd_results = (
        data
        .groupby(subject_col)[methods]
        .agg(lambda x: rpd(x))
        .reset_index()
    )
    return rpd_results

def compute_spearman_consistency(
    data, 
    methods=None, 
    start_col=None, 
    time_col='timepoint', 
    subject_col='subjectID', 
    tp0='TP0', 
    tp1='TP1', 
    n_iter=1000, 
    random_state=None
):
    """
    Compute Spearman rank correlation between TP0 and TP1 for each method,
    and perform null testing via permutation.

    Parameters:
    - data: DataFrame containing repeated measures
    - methods: list of method column names (optional)
    - start_col: column index to start method columns from (optional)
    - time_col: column name indicating timepoints
    - subject_col: column name for subject IDs
    - tp0, tp1: labels for the two timepoints
    - n_iter: number of permutations for null testing
    - random_state: random seed

    Returns:
    - DataFrame with columns: method, spearman_corr, p_value
    """
    if methods is None:
        if start_col is None:
            raise ValueError("Either 'methods' or 'start_col' must be provided.")
        methods = list(data.columns[start_col:])

    if random_state is not None:
        np.random.seed(random_state)

    # Filter and sort by timepoint and subject
    tp0_df = data[data[time_col] == tp0].sort_values(subject_col)
    tp1_df = data[data[time_col] == tp1].sort_values(subject_col)

    results = []

    for method in methods:
        x = tp0_df[method].values
        y = tp1_df[method].values

        # Check equal length
        if len(x) != len(y):
            raise ValueError(f"Mismatch in number of subjects for method {method} between {tp0} and {tp1}")

        # Actual Spearman correlation
        corr, _ = stats.spearmanr(x, y)

        # Null distribution
        null_corrs = []
        for _ in range(n_iter):
            y_shuffled = np.random.permutation(y)
            null_corr, _ = stats.spearmanr(x, y_shuffled)
            null_corrs.append(null_corr)

        # Two-sided p-value
        null_corrs = np.array(null_corrs)
        p_value = np.mean(np.abs(null_corrs) >= abs(corr))

        results.append({
            'method': method,
            'spearman_corr': corr,
            'p_value': p_value
        })

    return pd.DataFrame(results)

def batch_effect_lrt(data, z_method, batch_col, time_col, age_col, random_effect):
    """
    Perform likelihood ratio test for batch effect
    """
    try:
        formula_full = f"{z_method} ~ {age_col} + C({time_col}) + C({batch_col})"
        formula_reduced = f"{z_method} ~ {age_col} + C({time_col})"
        
        # Fit models with error handling
        md_full = smf.mixedlm(formula_full, data, groups=data[random_effect]).fit(reml=False)
        md_reduced = smf.mixedlm(formula_reduced, data, groups=data[random_effect]).fit(reml=False)
        
        # Calculate likelihood ratio test
        lr_stat = 2 * (md_full.llf - md_reduced.llf)
        df_diff = md_full.df_modelwc - md_reduced.df_modelwc
        
        # Ensure df_diff is positive
        if df_diff <= 0:
            print(f"Warning: df_diff = {df_diff}, setting to 1")
            df_diff = 1
            
        p_value = stats.chi2.sf(lr_stat, df_diff)
        return p_value
        
    except Exception as e:
        print(f"Error in batch_effect_lrt: {e}")
        return np.nan

def pairwise_site_tests(model, group_var, data):
    """
    Perform pairwise tests between sites/batches
    """
    try:
        fixed_effects_names = model.fe_params.index
        site_terms = [term for term in fixed_effects_names if term.startswith(f"C({group_var})[T.")]
        site_names = [term.split('T.')[1].rstrip(']') for term in site_terms]

        # Ensure group_var is categorical
        if not isinstance(data[group_var].dtype, pd.CategoricalDtype):
            data[group_var] = data[group_var].astype('category')

        group_cats = data[group_var].cat.categories.tolist()

        # Determine reference site
        ref_site_candidates = list(set(group_cats) - set(site_names))
        if len(ref_site_candidates) == 1:
            ref_site = ref_site_candidates[0]
        else:
            ref_site = group_cats[0]  # fallback to first category

        all_sites = [ref_site] + site_names
        sig_count = 0
        total_comparisons = 0

        print(f"Reference site: {ref_site}")
        print(f"All sites: {all_sites}")
        print(f"Fixed effects: {list(fixed_effects_names)}")

        for site1, site2 in combinations(all_sites, 2):
            total_comparisons += 1
            contrast = np.zeros(len(fixed_effects_names))
            
            # Set up contrast vector
            if site1 != ref_site:
                idx1 = f"C({group_var})[T.{site1}]"
                if idx1 in fixed_effects_names:
                    contrast[fixed_effects_names.get_loc(idx1)] = 1
                    
            if site2 != ref_site:
                idx2 = f"C({group_var})[T.{site2}]"
                if idx2 in fixed_effects_names:
                    contrast[fixed_effects_names.get_loc(idx2)] -= 1
            
            # Skip if contrast is all zeros (comparing reference to itself)
            if np.allclose(contrast, 0):
                continue
                
            try:
                # Convert 1D contrast to 2D matrix (row vector) for t_test
                contrast_matrix = contrast.reshape(1, -1)
                print(f"Contrast matrix shape: {contrast_matrix.shape}")
                print(f"Model k_fe (number of fixed effects): {model.k_fe}")
                
                test_res = model.t_test(contrast_matrix)
                
                # Extract p-value
                pval = test_res.pvalue
                if isinstance(pval, np.ndarray):
                    pval = pval.flat[0]  # Get first element safely
                elif hasattr(pval, 'item'):
                    pval = pval.item()
                
                # Ensure pval is a scalar float
                pval = float(pval)
                
                if pval < 0.05:
                    sig_count += 1
                    print(f"Significant difference between {site1} and {site2}: p = {pval:.4f}")
                else:
                    print(f"Non-significant difference between {site1} and {site2}: p = {pval:.4f}")
                    
            except Exception as e:
                print(f"Error testing {site1} vs {site2}: {e}")
                print(f"Contrast vector shape: {contrast.shape}")
                print(f"Model fixed effects count: {getattr(model, 'k_fe', 'unknown')}")
                print(f"Fixed effects names: {list(fixed_effects_names)}")
                continue

        print(f"Found {sig_count} significant pairwise differences out of {total_comparisons} comparisons")
        return sig_count, total_comparisons
        
    except Exception as e:
        print(f"Error in pairwise_site_tests: {e}")
        import traceback
        traceback.print_exc()
        return np.nan, np.nan


def run_lme_diagnostics(
    data,
    methods=None,
    start_col=None,
    age_col="age",
    batch_col="batch",
    time_col="timepoint",
    random_effect="subjectID"
):
    """
    Run comprehensive LME diagnostics
    """
    if methods is None:
        if start_col is None:
            raise ValueError("Provide either 'methods' or 'start_col'.")
        methods = list(data.columns[start_col:])

    data = data.copy()
    warnings.warn("Age and method values will be z-scored.")
    
    # Z-score age
    data['z_age'] = zscore(data[age_col])
    
    # Ensure categorical variables are properly set
    data[batch_col] = data[batch_col].astype("category")
    data[time_col] = data[time_col].astype("category")
    
    # Print diagnostic information
    print(f"Number of batches: {data[batch_col].nunique()}")
    print(f"Batch categories: {data[batch_col].cat.categories.tolist()}")
    print(f"Number of timepoints: {data[time_col].nunique()}")
    print(f"Timepoint categories: {data[time_col].cat.categories.tolist()}")
    print(f"Number of subjects: {data[random_effect].nunique()}")

    results = []

    for i, method in enumerate(methods):
        print(f"\nProcessing method {i+1}/{len(methods)}: {method}")
        
        z_method = f"z_{method}"
        data[z_method] = zscore(data[method])
        row = {"method": method}

        # --- Model 1: Full model for batch effect and pairwise test ---
        try:
            formula1 = f"{z_method} ~ z_age + C({time_col}) + C({batch_col})"
            print(f"Fitting model: {formula1}")
            model1 = smf.mixedlm(formula1, data, groups=data[random_effect]).fit()
            
            # Batch effect LRT
            batch_p = batch_effect_lrt(data, z_method, batch_col, time_col, 'z_age', random_effect)
            row["batch_effect_p"] = batch_p
            print(f"Batch effect p-value: {batch_p}")
            
            # Pairwise site tests
            sig_sites_n, total_comps = pairwise_site_tests(model1, batch_col, data)
            row["n_sig_pairwise_sites"] = sig_sites_n
            row["total_pairwise_comparisons"] = total_comps
            
        except Exception as e:
            print(f"Error in Model 1: {e}")
            row["batch_effect_p"] = np.nan
            row["n_sig_pairwise_sites"] = np.nan
            row["total_pairwise_comparisons"] = np.nan
            row["error_model1"] = str(e)

        # --- Model 2: ICC estimation ---
        try:
            formula2 = f"{z_method} ~ 1"
            model2 = smf.mixedlm(formula2, data, groups=data[random_effect]).fit()
            var_between = model2.cov_re.iloc[0, 0]
            var_within = model2.scale
            icc = var_between / (var_between + var_within)
            row["between_subject_var"] = var_between
            row["residual_error_var"] = var_within
            row["ICC"] = icc
            row["resid_to_between_ratio"] = var_within / var_between
        except Exception as e:
            print(f"Error in Model 2: {e}")
            row["between_subject_var"] = np.nan
            row["residual_error_var"] = np.nan
            row["ICC"] = np.nan
            row["resid_to_between_ratio"] = np.nan
            row["error_model2"] = str(e)

        # --- Model 3: Age and timepoint effects ---
        try:
            formula3 = f"{z_method} ~ z_age + C({time_col})"
            model3 = smf.mixedlm(formula3, data, groups=data[random_effect]).fit()

            # Age effect
            age_param = "z_age"
            if age_param in model3.params:
                row[f"{age_param}_estimate"] = model3.params[age_param]
                row[f"{age_param}_p"] = model3.pvalues[age_param]
                ci = model3.conf_int().loc[age_param]
                row[f"{age_param}_CI_lower"] = ci[0]
                row[f"{age_param}_CI_upper"] = ci[1]
            
            # Timepoint effect (using the second category as reference)
            time_cats = data[time_col].cat.categories
            if len(time_cats) > 1:
                time_param = f"C({time_col})[T.{time_cats[1]}]"
                if time_param in model3.params:
                    row[f"{time_param}_estimate"] = model3.params[time_param]
                    row[f"{time_param}_p"] = model3.pvalues[time_param]
                    ci = model3.conf_int().loc[time_param]
                    row[f"{time_param}_CI_lower"] = ci[0]
                    row[f"{time_param}_CI_upper"] = ci[1]
                    
        except Exception as e:
            print(f"Error in Model 3: {e}")
            row["error_model3"] = str(e)

        results.append(row)

    return pd.DataFrame(results)