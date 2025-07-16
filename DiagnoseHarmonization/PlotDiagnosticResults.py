#%%
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Sequence
import numpy as np
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for Cohens D results ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
def CohensD(cohens_d, df: None) -> list:
    """
    Plots the output of Cohens D as a bar plot with a histogram of the values on different axes at the same scale.

    Args:
        result (pd.DataFrame): DataFrame containing the results of Cohens D.
        doall_df (pd.DataFrame, optional): DataFrame containing additional data for plotting. Defaults to None.

    Returns:
        None: Displays the plot.
    """
    import matplotlib.pyplot as plt
    from collections.abc import Sequence

    if df is not None:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Dataframe must be a pandas DataFrame")
        if 'CohensD' not in df.columns:
            raise ValueError("Dataframe must contain a 'CohensD' column")
        
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    # Dummy data (replace with your actual data)
    np.random.seed(0)
    cohens_d = np.random.normal(0, 0.5, 100)

    # Set up figure and gridspec
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 8], wspace=0.3)

    # Histogram on the left
    ax1 = fig.add_subplot(gs[0])
    ax1.hist(cohens_d, bins=20, orientation='horizontal', color=[0.8, 0.2, 0.2])
    ax1.set_xlabel("Proportion")
    ax1.invert_xaxis()
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")

    # Bar plot on the right
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    indices = np.arange(len(cohens_d))
    bars = ax2.bar(indices, cohens_d, color=[0.2, 0.4, 0.6])
    ax2.plot(indices, cohens_d, 'r.')

    # Significance lines
    effect_sizes = [
        (0.2, 'Small effect size', 'g'),
        (0.5, 'Medium effect size', 'b'),
        (0.8, 'Large effect size', 'r'),
        (2.0, 'Huge effect size', 'm')
    ]

    for val, label, color in effect_sizes:
        ax2.axhline(y=val, linestyle='--', color=color, label=label)
        ax2.axhline(y=-val, linestyle='--', color=color)

    # Labels and grid
    ax2.set_xlabel("IDP index")
    ax2.set_ylabel("Cohen's d: $(\\mu_1 - \\mu_2)/\\sigma_{pooled}$")
    ax2.set_title("Effect Size (Cohen's d) for T2 Batch Effect Across Structural IDPs")
    ax2.grid(True)
    #plt.tight_layout()
    plt.show()
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for PCA correlation results ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""

def Plot_PC_corr(PrincipleComponents, batch, covariates=None, variable_names=None, PC_correlations = False):
    """
    Plots the first two PCs as a scatter plot with batch indicated by color.
    parameters:
        PrincipleComponents (np.ndarray): The PCA scores (subjects x N_components).
        batch (np.ndarray): Subjects x 1, batch labels.
        covariates (np.ndarray, optional): Subjects x covariates, additional variables to correlate with PCs. Defaults to None.
        variable_names (list of str, optional): Names for the variables. Defaults to None.
    Returns:
        None: Displays the plot.
    Raises:
        ValueError: If PrincipleComponents is not a 2D array or batch is not a
        1D array, or if the number of samples in PrincipleComponents and batch do not match.

    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    # Check number of batches
    unique_batches = np.unique(batch)
    if len(unique_batches) < 2:
        raise ValueError("At least two unique batches are required")

    # iteratvely plot the first two PCs, seperated by batch
    import matplotlib.pyplot as plt

    if variable_names is None:
        if covariates is not None:
            variables = np.column_stack((batch, covariates))
            variable_names = ['Batch'] + [f'Covariate{i+1}' for i in range(covariates.shape[1])]
        else:
            variables = batch
            variable_names = [f"Batch"]  
    # Create a DataFrame for plotting

    import pandas as pd

    PC_Names = [f"PC{i+1}" for i in range(PrincipleComponents.shape[1])]
    df = pd.DataFrame(PrincipleComponents, columns=PC_Names[:PrincipleComponents.shape[1]])
    df['batch'] = batch

    if covariates is not None:
        for i in range(covariates.shape[1]):
            df[f'Covariate{i+1}'] = covariates[:, i]

    # Plotting by batch
    plt.figure(figsize=(10, 8))
    for i in range(len(unique_batches)):
        batch_data = df[df['batch'] == unique_batches[i]]
        #plt.scatter(batch_data[variable_names[0]], batch_data[variable_names[1]], label=f'Batch {unique_batches[i]}', alpha=0.6)
        # Plotting the first two PCs as a scatter plot
        plt.scatter(PrincipleComponents[batch == unique_batches[i], 0],
                    PrincipleComponents[batch == unique_batches[i], 1],
                    label=f'Batch {unique_batches[i]}', alpha=0.6)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Scatter Plot by Batch')
        plt.legend()
        plt.grid(True)
    plt.show()

    # Plotting by covariates if provided
    if covariates is not None:
        for i in range(covariates.shape[1]):
            plt.figure(figsize=(10, 8))
            # Check if covariate is continuous or categorical, as categorical may be binary, check by number of unique values
            if len(np.unique(covariates[:, i])) <= 20:  # Assuming
                # If categorical, use a scatter plot of first two PCs, with discrete colours for each category indicated in the legend
                unique_categories = np.unique(covariates[:, i])
                for category in unique_categories:
                    category_data = df[df[f'Covariate{i+1}'] == category]
                    # Plotting the first two PCs as a scatter plot by covariate category
                    plt.scatter(PrincipleComponents[category_data.index, 0],
                                PrincipleComponents[category_data.index, 1],
                                label=f'{variable_names[i+1]} = {category}', alpha=0.6)
                    
            elif np.issubdtype(covariates[:, i].dtype, np.number):  # Check if continuous
                # If continous, use a scatter plot of first two PCs, with opacity based on covariate value
                plt.scatter(PrincipleComponents[:, 0], PrincipleComponents[:, 1],
                            c=covariates[:, i], cmap='viridis', alpha=0.6, label=f'{variable_names[i+1]} {i+1}')
                plt.colorbar(label=f'{variable_names[i+1]}{i+1}')
            else:
                raise ValueError(f"Covariate {i+1} must be either continuous or categorical, got {covariates[:, i].dtype}")
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title(f'PCA Scatter Plot by Covariate {i+1}')
            plt.legend()
            plt.grid(True)
            plt.show()  

    # Calculate and plot correlations with PCs if PC_correlations is True
    if PC_correlations:
        if covariates is None:
            raise Warning("Covariates not provided proceeding with just batch correlation")
            correlations = np.corrcoef(PrincipleComponents.T, batch.T)[:PrincipleComponents.shape[1], PrincipleComponents.shape[1]:]
        else:
            # Calculate correlations between PCs, covariates and batch
            if not isinstance(covariates, np.ndarray):
                raise ValueError("Covariates must be a numpy array")
        # Combine batch, covariates and PCS into a single array for correlation
            combined_data = np.column_stack((PrincipleComponents, batch, covariates))
            # Combine names for axes
            combined_variable_names = variable_names + [f'PC{i+1}' for i in range(PrincipleComponents.shape[1])]
            # Calculate correlations
            correlations = np.corrcoef(combined_data.T)
        # Plot the correlation matrix
        import seaborn as sns
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlations, annot=True, fmt=".2f", cmap='coolwarm',
                     xticklabels=combined_variable_names, yticklabels=combined_variable_names)
        plt.title('Correlation Matrix of PCs, Batch and Covariates')
        plt.show()    
# %%
