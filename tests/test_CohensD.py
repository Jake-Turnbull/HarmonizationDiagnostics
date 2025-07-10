from DiagnoseHarmonization import DiagnosticFunctions

import numpy as np

group1 = np.array([1,2,3,4,5])
group2 = np.array([2,3,4,5,6])

def test_cohens_d():
    assert DiagnosticFunctions.Cohens_D(group1, group2) == [0, 0, 0, 0, 0]
