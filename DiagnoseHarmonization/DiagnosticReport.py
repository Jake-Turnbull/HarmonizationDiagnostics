import argparse
parser=argparse.ArgumentParser(
    description='''Function which performs a variety of diagnostic tests on a dataset to assess harmonization necessity,
    quality and potential issues''',
)
#parser.add_argument('--', type=int, default=42, help='FOO!')
#parser.add_argument('bar', nargs='*', default=[1, 2, 3], help='BAR!')
args=parser.parse_args()

import numpy as np
import pandas as pd
import os
from datetime import datetime



def FullReport(data: np.ndarray, batch, log_path: str = None):
    import numpy as np
    import pandas as pd
    import os
    from datetime import datetime
    import logging
    """
    
    Args:
        data (np.ndarray): A 2D array with shape (subjects, features).

        batch (list or np.ndarray): A 1D array of group labels (string or numeric).
        log_location (str, optional): Path to save log file. Defaults to current working directory.

    Returns:
        pd.DataFrame: A DataFrame with features and group labels.

    """

    # Setup logging (this will run once)
    def setup_logger(log_path=None):
        if log_path is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_filename = f"Diagnostic_Log_{timestamp}.txt"
            log_location = os.getcwd()  # Default to current working directory
            log_path = os.path.join(log_location, log_filename)

        logging.basicConfig(
            filename=log_path,
            filemode='w',  # overwrite each time; use 'a' to append
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        # Optional: log to console too
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logging.getLogger('').addHandler(console)

        logging.info(f"Logging started. Output file: {log_path}")
    setup_logger(log_path=log_path)


    if isinstance(batch, (list, np.ndarray)):
        batch = np.array(batch)
        if batch.dtype.kind in {'U', 'S', 'O'}:  # string or object (categorical)
            logging.info("Converting categorical batch to numeric codes")
            batch, unique = pd.factorize(batch)
            logging.info(f"Batch categories: {list(unique)}")
    else:
        raise ValueError("Batch must be a list or numpy array")
    logging.info("Batch converted to numeric codes.")

    # Input validation
    if data.shape[0] != len(batch):
        raise ValueError("Length of batch must match number of subjects (rows in data).")
        logging.error("Input validation failed: Length of batch does not match number of subjects.")
    
    logging.info("Input validation passed.")
    # Construct DataFrame
    df = pd.DataFrame(data, columns=[f"Feature_{i+1}" for i in range(data.shape[1])])
    df["Group"] = batch
    
    logging.info("DataFrame constructed with features and group labels.")


    log_start = (
        f"Analysis started: {datetime.now()}\n"
        f"Number of subjects: {data.shape[0]}\n"
        f"Number of features: {data.shape[1]}\n"
        f"Unique batch: {set(batch)}\n"
        f"Log location: {log_path}\n"
    )
    logging.info("Starting diagnostic report generation.")
    logging.info(log_start)

    logging.info("Loading diagnostic functions from DiagnoseHarmonization module" \
    "----------------------------------------------------------------------------")
    # Import diagnostic functions
    from DiagnoseHarmonization import DiagnosticFunctions
    # Perform diagnostics

    logging.info("Successfully loaded diagnostic functions from DiagnoseHarmonization module")

    logging.info("Performing Cohen's d calculation.")
    cohens_d = DiagnosticFunctions.Cohens_D(data, data)  # Using data

    logging.info("Cohen's d calculation completed.")

    # Add Cohen's d results to DataFrame

    df[f"Cohen's_d"] =  cohens_d
    logging.info(f"Cohen's d results added to DataFrame, " \
    f"average Cohen's d for across features is : {np.mean(cohens_d)}" \
    f" with a variance of {np.var(cohens_d)}")

    logging.info("Performing PCA correlation.")

    pearsonr, explained_variance, score, batchPCcorr = DiagnosticFunctions.PcaCorr(data, batch)
    logging.info("PCA correlation completed.")
    logging.info(f"Pearson correlation of batch with first 3 Principal components results: {batchPCcorr}")
    logging.info(f"Explained variance for each PC: {explained_variance}")
    

    # Add PCA results to DataFrame
    for i in range(len(explained_variance)):            
        df[f"PC_{i+1}"] = score[:, i]
    #df["BatchPCcorr"] = batchPCcorr 
    df["Pearson_Corr_Batch_PC's"] = pearsonr
    print(pearsonr)
    logging.info("PCA results added to DataFrame.")

    return df

# Example data
import numpy as np

data = np.random.rand(5, 3) # 5 subjects, 3 features
groups = ['A', 'B', 'A', 'B', 'A']

df = FullReport(data, groups,log_path='diagnostic_report.txt')

#print(df)

