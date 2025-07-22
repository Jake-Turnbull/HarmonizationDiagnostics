import pandas as pd
from DiagnoseHarmonization.DiagnosticFunctions import run_lme_diagnostics

all_data = pd.read_csv('/Users/psyc1586_admin/GVB_data/harmonisation_work/python_codes/allData.csv')
print(all_data)

# Assuming all_data is dataframe
model_results = DiagnosticFunctions.run_lme_diagnostics(
    data=all_data,
    start_col=7,
    age_col='age',
    batch_col='batch',
    time_col='timepoint', 
    random_effect='subjectID'
)
print(model_results)
model_results.to_csv("lme_diagnostics_results.csv", index=False)
