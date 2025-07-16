#%%
from DiagnoseHarmonization import DiagnosticFunctions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

group1 = np.array([1,2,3,4,5])
group2 = np.array([2,3,4,5,6])

def test_cohens_d():
    assert DiagnosticFunctions.Cohens_D(group1, group2) == [0, 0, 0, 0, 0]

#%%
from DiagnoseHarmonization import PlotDiagnosticResults
"""
def test_cohens_d_plot():
    cohens_d = DiagnosticFunctions.Cohens_D(group1, group2)
    df = pd.DataFrame({'CohensD': cohens_d})
    
    # Test if the function runs without errors
    try:
        PlotDiagnosticResults.CohensD(cohens_d,df=None)
    except Exception as e:
        assert False, f"CohensD function raised an exception: {e}"
    
    # Check if the plot is created (this is a basic check)
    #assert plt.fignum_exists(1), "Plot was not created"
"""
# %%
