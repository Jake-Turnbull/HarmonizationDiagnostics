""" Test script for ComBat

Simulate a small dummy dataet and run the ComBat harmonization on it, printing the result in terminal.

Dataset structure is constructed to match the expectaions of ComBat

Y = alpha + X*beta_age +X*beta_sex + gamma_batch + delta_batch*epsilon

Here we simulate the dataset with two batches, one continuous covariate (age) and one binary (sex)

The data consists of 10 features (rows) and 20 samples (columns)

We create an array of random data of size feature x samples (10x20) drawn from a standard normal distribution centered at zero for each feature
, a feature mean of size (10x1), two covariates of size (20x1) respectively,

We then simulate a multiplicative batch effect (delta) and an additive batch effect (gamma)

The random array is multipled by the multiplicative batch effect and we then add the feature mean, covariate effects 
and additive batch effect to create the final dataset.


Here we can consider the final dataset to be the observed data convered to Z-scores for which we apply ComBat.
If ComBat runs correctly, we should see no errors.

"""


import numpy as np
from DiagnoseHarmonization import HarmonizationFunctions

# Simulate a small datset that is normally distributed

subject_data = np.random.normal(0, 1, (10, 20))  # 10 features, 20 samples

# Simulate a feature mean to be randomly distributed around 0, the standard deviation is 1 here due to the z-scoring of the data

feature_mean = np.random.normal(0, 1, (10, 1))  # 10 features, 1 mean per feature

# Simulate two covariates, one continuous (age) and one binary
age = np.random.randint(50, 70, size=(20, 1))  # Continuous covariate (age)
# Define a beta coefficient for age for each feature, this will be multiplied by mean centered age covariate
beta_age = np.random.normal(0, 0.1, (10, 1))  # Small effect of age on each feature

sex = np.random.randint(0, 2, size=(20, 1))  # Binary cov
# Combine covariates into a design matrix
covariates = np.column_stack((age, sex)) # Shape (20, 2)
# Repeat above step for the sex coefficient, mean centering again to avoid intercept issues and multiply by 0
beta_sex = np.random.normal(0, 0.1, (10, 1)) 

# Simulate batch variable with two batches
batch = np.array([0]*10 + [1]*10)  # 20 samples

# Additive batch effect will be small and normally distributed
gamma = np.random.normal(0, 0.5, (10, 1)) # Setting it to 0.5 means the worst effect causes the data to shift by 0.5 standard deviations

# The delta batch effect is assumed to be from an inverse gamma distribution
delta = np.random.uniform(0.5, 1.5, (10, 1)) # Setting it to between 0.5 and 1.5 means the random subject effect can be multipled between 0.5 and 1.5

# Create the final dataset
# Y = alpha + X*beta_age +X*beta_sex + gamma_batch + delta_batch*epsilon

final_data = (subject_data * delta) + feature_mean + (beta_age @ (age - np.mean(age)).T) + (beta_sex @ (sex - np.mean(sex)).T) + gamma @ (batch - np.mean(batch)).reshape(1, -1)

# Print the final dataset
print("Final simulated dataset (features x samples):")
print(final_data)

# Apply ComBat
# Transpose the data to match ComBat expectations (samples x features)

harmonized_data, delta_star, gamma_star = HarmonizationFunctions.combat_modified(final_data, batch, covariates, parametric=True, DeltaCorrection=True, UseEB=True, ReferenceBatch=None, RegressCovariates=False, GammaCorrection=True)
print("Harmonized data (samples x features):")
