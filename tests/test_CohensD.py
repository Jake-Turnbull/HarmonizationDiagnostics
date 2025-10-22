#%%
from DiagnoseHarmonization import DiagnosticFunctions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

group1 = np.array([1,2,3,4,5])
group2 = np.array([1,2,3,4,5])
group = np.concatenate((group1, group2))
batch = np.array([0,0,0,0,0,1,1,1,1,1])
# Convert batch to numeric codes

#%%

def test_cohens_d():
    group = np.random.rand(10,100)
    batch = np.array([0,0,0,0,0,1,1,1,1,1])
    a,b = DiagnosticFunctions.Cohens_D(group, batch)
    assert np.size(a) ==100
    print("Tested Cohen's D function successfully.")

#%%
import random
import numpy as np
from DiagnoseHarmonization import DiagnosticFunctions

group = np.random.rand(15,100)
batch = np.array([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2])
print(np.shape(group))
print(np.shape(batch))
(cohens_d, labels) = DiagnosticFunctions.Cohens_D(group, batch)
print(np.shape(cohens_d))

from DiagnoseHarmonization import PlotDiagnosticResults
PlotDiagnosticResults.Cohens_D_plot(cohens_d,labels)
print("Plotted Cohen's D successfully.")
plt.close("all")


#%%
from DiagnoseHarmonization import PlotDiagnosticResults

def test_cohens_d_plot():
    from DiagnoseHarmonization import DiagnosticFunctions
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    group = np.random.rand(10,100)
    batch = np.array([0,0,0,0,0,1,1,1,1,1])
    (cohens_d, labels) = DiagnosticFunctions.Cohens_D(group, batch)
    #df = pd.DataFrame({'CohensD': cohens_d})
    
    # Test if the function runs without errors
    try:
        PlotDiagnosticResults.Cohens_D_plot(cohens_d,labels,df=None)
    except Exception as e:
        assert False, f"CohensD function raised an exception: {e}"

    # Check if the plot is created (this is a basic check)
    #assert plt.fignum_exists(1), "Plot was not created"
    #return df
    print("Tested Cohen's D plot function with multiple batches successfully.")
    plt.close("all")
#%%
