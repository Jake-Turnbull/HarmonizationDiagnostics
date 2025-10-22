"""
Script containing self contained harmonization functions that can be used in conjunction with the diagnostic tools:

ComBat:
    Run ComBat harmonization on the data and return the harmonized data
    Parameters:

    dat: numpy array of shape (n_features, n_samples)
        or pandas DataFrame of shape (n_samples, n_features)
        The data to be harmonized, where rows are features and columns are samples.

    batch: numpy array of shape (n_samples,1)
        or pandas Series of shape (n_samples,1)
        The batch variable indicating the batch membership for each sample.

    mod: Numpy array of shape (n_samples, n_covariates)
        or pandas DataFrame of shape (n_samples, n_covariates)
        The design matrix for the covariates, these can either be preserved or treated as confounds and removed from the data.

    parametric: bool, optional (default=True)
        Whether to use parametric adjustments (True) or non-parametric adjustments (False).

    DeltaCorrection: bool, optional (default=True)
        Whether to correct for additive batch effects (True) or not (False).

    GammaCorrection: bool, optional (default=True)
        Whether to correct for multiplicative batch effects (True) or not (False).

    UseEB: bool, optional (default=True)
        Whether to use Empirical Bayes to estimate the batch effect parameters (True) or not
        (False).

    RemoveConfounders: bool, optional (default=False)
        Whether to treat covariates as confounds and remove their effects from the data (True
        or to preserve covariate effects in the data (False).

    ReferenceBatch: int or str, optional (default=None)
        The batch to use as the reference batch for harmonization. If None, all batches are used
        to estimate harmonization parameters.
"""


"""
Line-by-line Python match of the provided MATLAB `combat_modified`.

Notes:
- This file intentionally mirrors the MATLAB control flow and variable names
  so you can paste in your own `aprior`, `bprior`, `itSol` implementations.
- Replace the stub functions `aprior`, `bprior`, `itSol` with your MATLAB-matched
  Python versions (I left them as placeholders so you could provide the exact
  code you want).

Usage:
    from combat_modified_matched import combat_modified
    bayesdata, delta_star, gamma_star = combat_modified(dat, batch, mod, parametric=True,
                                                        DeltaCorrection=True, UseEB=True,
                                                        ReferenceBatch=None,
                                                        RegressCovariates=False, GammaCorrection=True)

Input shapes match the MATLAB expectations:
    dat : (n_features, n_samples) numpy array
    batch: (n_samples,) vector of batch labels (integers or categories)
    mod: (n_samples, n_covariates) or None

"""

import numpy as np
import pandas as pd

# --------------------- Placeholder helper functions ---------------------
# Translated from MATLAB, need to have concistency checked with NeuroComBat

def aprior(delta_hat):
    m = np.mean(delta_hat)
    s2 = np.var(delta_hat,ddof=1)
    return (2 * s2 +m**2) / float(s2)

def bprior(delta_hat):
    m = delta_hat.mean()
    s2 = np.var(delta_hat,ddof=1)
    return (m*s2+m**3)/s2

def postmean(g_hat, g_bar, n, d_star, t2):
    return (t2*n*g_hat+d_star * g_bar) / (t2*n+d_star)

def postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


def itSol(sdat_batch, gamma_hat, delta_hat, gamma_bar, t2, a, b, conv=0.001):
    g_old = gamma_hat
    d_old = delta_hat
    change = 1
    count = 0
    while change > conv:
        g_new = postmean(gamma_hat, gamma_bar, sdat_batch.shape[1], d_old, t2)
        sum2 = np.sum((sdat_batch - g_new[:, None]) ** 2, axis=1)
        d_new = postvar(sum2, sdat_batch.shape[1], a, b)
        change = max(np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-8)),
                     np.max(np.abs(d_new - d_old) / (np.abs(d_old) + 1e-8)))
        g_old = g_new
        d_old = d_new
        count += 1
        if count > 100:
            print('Warning: itSol did not converge after 100 iterations')
            break
    return np.vstack([g_new, d_new])

# ----------------------------- Main function -----------------------------

def combat_modified(dat, batch, mod, parametric,
                    DeltaCorrection=True, UseEB=True, ReferenceBatch=None,
                    RegressCovariates=False, GammaCorrection=True):
    """
    Python line-by-line translation of the MATLAB combat_modified provided by user.

    Returns:
        bayesdata : (n_features, n_samples) corrected data
        delta_star : (n_batch, n_features)
        gamma_star : (n_batch, n_features)
    """

    # Make sure inputs are numpy arrays
    dat = np.asarray(dat, dtype=float)
    batch = np.asarray(batch)

    # Option messages (mirrors MATLAB display behaviour)
    if ReferenceBatch is None:
        print('Reference batch not given, defaulting to no reference')
    else:
        print(f'ReferenceBatch = {ReferenceBatch} -- fitting prior estimates using this batch and leaving batch unchanged')

    if not UseEB:
        print('Empirical Bayes set to false, using first estimates from raw mean and variances')
    else:
        print('Empirical Bayes set to true')



    if RegressCovariates:
        print('Regress Covariates set to true, skipping re-addition of OLS covariate estimates ')

    if not DeltaCorrection:
        print('Delta correction set to False, applying no delta (scale) correction on data')

    if not GammaCorrection:
        print('Gamma correction set to False, applying no gamma (mean) correction on data')

    # --------------------- Begin ComBat core logic ---------------------

    # Compute SDs across samples for each feature (row)
    sds = np.std(dat, axis=1, ddof=1)  # MATLAB std(...')' used sample std
    wh = np.where(sds == 0)[0]
    if wh.size > 0:
        raise ValueError('Error. There are rows with constant values across samples. Remove these rows and rerun ComBat.')

    # Convert batch vector to categorical and create dummy variables
    batch_cat = pd.Categorical(batch)
    batchmod = pd.get_dummies(batch_cat, drop_first=False).values  # shape (n_samples, n_batch)

    # Number of batches
    n_batch = batchmod.shape[1]
    levels = np.array(batch_cat.categories)
    print(f'[combat] Found {n_batch} batches')

    # Create list of arrays each containing sample indices for a batch
    batches = [np.where(batch == lev)[0] for lev in levels]

    # Size of each batch and total number of samples
    n_batches = np.array([len(b) for b in batches])
    n_array = np.sum(n_batches)

    # Construct design matrix including batch and additional covariates (mod)
    if mod is None:
        mod_arr = np.zeros((dat.shape[1], 0))
    else:
        mod_arr = np.asarray(mod, dtype=float)
        if mod_arr.ndim == 1:
            mod_arr = mod_arr.reshape(-1, 1)

    design = np.hstack([batchmod, mod_arr])  # shape (n_samples, n_batch + n_cov)

    # Remove intercept column if present
    intercept = np.ones((n_array, 1))
    cols_to_keep = []
    for j in range(design.shape[1]):
        if not np.allclose(design[:, j], intercept.ravel()):
            cols_to_keep.append(j)
    design = design[:, cols_to_keep]

    print(f'[combat] Adjusting for {design.shape[1] - n_batch} covariate(s) of covariate level(s)')

    # Check for confounding between batch and covariates
    if np.linalg.matrix_rank(design) < design.shape[1]:
        nn = design.shape[1]
        if nn == (n_batch + 1):
            raise ValueError('Error. The covariate is confounded with batch. Remove the covariate and rerun ComBat.')
        if nn > (n_batch + 1):
            temp = design[:, (n_batch):nn]
            if np.linalg.matrix_rank(temp) < temp.shape[1]:
                raise ValueError('Error. The covariates are confounded. Please remove one or more of the covariates so the design is not confounded.')
            else:
                raise ValueError('Error. At least one covariate is confounded with batch. Please remove confounded covariates and rerun ComBat.')

    print('[combat] Standardizing Data across features')

    # Estimate coefficients B_hat using least squares: B_hat = inv(design' * design) * design' * dat'
    XtX = design.T @ design
    inv_XtX = np.linalg.pinv(XtX)
    B_hat = inv_XtX @ design.T @ dat.T  # shape (k, n_features)

    # Reference batch handling: use only reference batch to compute pooled mean/var when provided
    if ReferenceBatch is not None:
        # find index of reference batch among levels
        try:
            ref_idx = int(np.where(levels == ReferenceBatch)[0][0])
        except Exception:
            raise ValueError('ReferenceBatch not found in batch levels.')

        ref_samples = batches[ref_idx]
        # unadjusted reference-batch mean from B_hat:
        ref_batch_effect = B_hat[ref_idx, :]  # shape (n_features,)

        # If covariates exist, add their effect back in (use only design structure)
        if design.shape[1] > n_batch:
            tmp = design.copy()
            tmp[:, :n_batch] = 0
            Cov_effects = (tmp @ B_hat).T  # shape (n_features, n_samples)
        else:
            Cov_effects = np.zeros((dat.shape[0], dat.shape[1]))

        # Build a design matrix just for the reference batch and compute residuals there
        design_ref = design[ref_samples, :]
        predicted_ref = (design_ref @ B_hat).T  # features x n_ref
        residuals_ref = dat[:, ref_samples] - predicted_ref  # features x n_ref

        # Compute variance across reference samples (pooled reference variance per feature)
        var_ref = np.mean(residuals_ref ** 2, axis=1)  # (n_features,)

        # replicate ref mean & variance across all samples
        stand_mean = np.tile(ref_batch_effect[:, None], (1, n_array))
        stand_mean = stand_mean + Cov_effects

        var_pooled = var_ref.copy()  # (n_features,)

        print(f'The size of the var_pooled array is {var_pooled.shape}')

    else:
        # No reference: pooled across all samples (original ComBat behaviour)
        n_features = dat.shape[0]
        n_samples = dat.shape[1]

        # recompute B_hat in case design changed (keeps parity with MATLAB)
        XtX = design.T @ design
        inv_XtX = np.linalg.pinv(XtX)
        B_hat = inv_XtX @ design.T @ dat.T

        # grand mean across batches
        grand_mean = (n_batches / n_array) @ B_hat[0:n_batch, :]  # shape (n_features,)

        # pooled variance across all samples (feature-wise)
        predicted = (design @ B_hat).T
        resid = dat - predicted
        var_pooled = np.mean(resid ** 2, axis=1)

        # Avoid zero pooled variances
        if np.any(var_pooled == 0):
            nonzeros = var_pooled[var_pooled != 0]
            if nonzeros.size > 0:
                var_pooled[var_pooled == 0] = np.median(nonzeros)
            else:
                var_pooled[var_pooled == 0] = 1e-6

        stand_mean = np.tile(grand_mean[:, None], (1, n_array))

        if design.shape[1] > n_batch:
            tmp = design.copy()
            tmp[:, :n_batch] = 0
            stand_mean = stand_mean + (tmp @ B_hat).T

    # Optional: regress covariates
    if design.shape[1] > n_batch:
        X_cov = design[:, n_batch:]
        X_cov = X_cov - np.mean(X_cov, axis=0, keepdims=True)
        B_cov = B_hat[n_batch:, :]
        Cov_effects = (X_cov @ B_cov).T
    else:
        Cov_effects = np.zeros_like(dat)

    # Standardize the data
    s_data = (dat - stand_mean) / (np.sqrt(var_pooled)[:, None] + 1e-8)

    # Estimate batch effect parameters using least squares
    print('[combat] Fitting L/S model and finding priors')
    batch_design = design[:, :n_batch]  # samples x n_batch
    XtX_b = batch_design.T @ batch_design
    inv_XtX_b = np.linalg.pinv(XtX_b)
    gamma_hat = inv_XtX_b @ batch_design.T @ s_data.T  # shape (n_batch, n_features)
    print(f'Size of gamma hat: {gamma_hat.shape}')

    # Estimate batch-specific variances
    delta_hat = np.zeros((n_batch, dat.shape[0]))
    for i in range(n_batch):
        indices = batches[i]
        if len(indices) > 1:
            delta_hat[i, :] = np.var(s_data[:, indices], axis=1, ddof=1)
        else:
            # if only 1 sample, variance is zero — fallback to small positive
            delta_hat[i, :] = np.var(s_data[:, indices], axis=1, ddof=0) + 1e-6

    print(f'Size of delta hat: {delta_hat.shape}')

    # Compute hyperparameters
    # gamma_bar and t2 are computed per-batch (matching MATLAB mean(gamma_hat') behaviour)
    gamma_bar = np.mean(gamma_hat, axis=1)  # (n_batch,)
    t2 = np.var(gamma_hat, axis=1, ddof=1)  # (n_batch,)
    t2[t2 == 0] = 1e-6

    # Compute a_prior, b_prior per batch using delta_hat rows
    a_prior = np.zeros(n_batch)
    b_prior = np.zeros(n_batch)
    for i in range(n_batch):
        a_prior[i] = aprior(delta_hat[i, :])
        b_prior[i] = bprior(delta_hat[i, :])

    # Apply empirical Bayes estimates (parametric)
    if parametric:
        print('[combat] Finding parametric adjustments')
        gamma_star = np.zeros_like(gamma_hat)
        delta_star = np.zeros_like(delta_hat)

        for i in range(n_batch):
            indices = batches[i]
            if len(indices) == 0:
                continue
            # s_data[:, indices] shape = (n_features x n_i)
            temp = itSol(s_data[:, indices], gamma_hat[i, :], delta_hat[i, :],
                         gamma_bar[i], t2[i], a_prior[i], b_prior[i], conv=0.001)
            # temp expected as 2 x n_features (row0 gamma, row1 delta)
            gamma_star[i, :] = temp[0, :]
            delta_star[i, :] = temp[1, :]

        # Reference batch: no correction (leave batch unchanged)
        if ReferenceBatch is not None:
            gamma_star[ref_idx, :] = np.zeros(dat.shape[0])
            delta_star[ref_idx, :] = np.ones(dat.shape[0])

    else:
        # If not parametric we use raw estimates (user can modify to non-parametric if desired)
        gamma_star = gamma_hat.copy()
        delta_star = delta_hat.copy()

    print('Size of gamma_star:', gamma_star.shape)

    # If UseEB == false in MATLAB script they set gamma_star/delta_star to raw estimates
    if not UseEB:
        print('Discounting the EB adjustments and using Raw estimates, this is not advised')
        delta_star = delta_hat.copy()
        gamma_star = gamma_hat.copy()

    # Apply the L/S adjustments to the standardized data
    print('[combat] Adjusting the Data')
    bayesdata = s_data.copy()
    j = 1

    if DeltaCorrection:
        if GammaCorrection:
            for i in range(n_batch):
                indices = batches[i]
                if len(indices) == 0:
                    continue
                # subtract gamma_star for that batch then divide by sqrt(delta_star)
                bayesdata[:, indices] = (bayesdata[:, indices] - (gamma_star[i, :])[:, None]) / (np.sqrt(delta_star[i, :])[:, None] + 1e-8)
                j += 1
        else:  # DeltaCorrection true, GammaCorrection false
            for i in range(n_batch):
                indices = batches[i]
                if len(indices) == 0:
                    continue
                bayesdata[:, indices] = bayesdata[:, indices] / (np.sqrt(delta_star[i, :])[:, None] + 1e-8)
                j += 1
    else:  # DeltaCorrection false
        if GammaCorrection:
            for i in range(n_batch):
                indices = batches[i]
                if len(indices) == 0:
                    continue
                bayesdata[:, indices] = (bayesdata[:, indices] - (gamma_star[i, :])[:, None])
                j += 1
        else:
            print('Warning: Both Gamma and delta have been set to false, no ComBat adjustments have been applied')

    # Transform data back to original scale
    if RegressCovariates:
        bayesdata = (bayesdata * (np.sqrt(var_pooled)[:, None])) + (stand_mean - Cov_effects)
    else:
        bayesdata = (bayesdata * (np.sqrt(var_pooled)[:, None])) + stand_mean

    return bayesdata, delta_star, gamma_star

# Define harmonization via mixed effects model (Regression analysis)

def lme_harmonization(data, batch, covariates, variable_names):

    # The function here is identical to that included in the DiagnosticFunctions.py file in methodology, but the 
    # Out put is instead the residualised data after removing the batch and optionally covariate effects.
    """
    Make mixed effect model including cross terms with batch and covariates,

    Parameters: 
        - Data: subjects x features (np.ndarray)
        - batch: subjects x 1 (np.ndarray), batch labels
        - covariates:  subjects x covariates (np.ndarray)
        - variable_names: covariates (list)
    Returns:
        - LME model results object
    Raises:
    - ValueError: if Data is not a 2D array or batch is not a
    1D array, or if the number of samples in Data and batch do not match.
    - ValueError: if covariates is not None and not a 2D array
    - ValueError: if variable_names is not None and does not match the number of variables
    """
    # Count the number of unique groups in the batch
    
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array (samples x features).")
    if not isinstance(batch, (list, np.ndarray)) or np.ndim(batch) != 1:
        raise ValueError("group_indices must be a 1D list or numpy array.")
    
    # Define the mixed effects model as Y = X*beta + e + Z*b
    # Where Y is the data, X is the design matrix, beta are the fixed effects
    # e is the residual error, Z is the random effects design matrix and b are the
    # random effects coefficients

    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
    import patsy
    import numpy as np
    import itertools
    import warnings
    warnings.filterwarnings("ignore")
    df = pd.DataFrame(data)
    df['batch'] = batch
    for i,var in enumerate(variable_names):
        df[var] = covariates[:,i]
    # Create interaction terms
    interaction_terms = []
    for var in variable_names:
        interaction_terms.append(f'batch:{var}')
    interaction_str = ' + '.join(interaction_terms)
    # Create the formula for the mixed effects model
    formula = f'Q("0") ~ batch + {" + ".join(variable_names)} + {interaction_str}'
    # Fit the mixed effects model
    model = mixedlm(formula, df, groups=df['batch'])
    result = model.fit()

    # Print the summary of the model
    print(result.summary())
    
    # Residualize the data by removing the batch and covariate effects
    residuals = result.resid.values
    return residuals


    