def test_variance_ratio_two_batches():
    from DiagnoseHarmonization import DiagnosticFunctions
    import numpy as np
    import matplotlib.pyplot as plt
    import pprint

    group = np.random.rand(10,100)
    batch = np.array([0,0,0,0,0,1,1,1,1,1])
    variance_ratio = DiagnosticFunctions.Variance_Ratios(group, batch)
    pprint.pprint(variance_ratio)

test_variance_ratio_two_batches()

def test_variance_ratio_multiple_batches():
    from DiagnoseHarmonization import DiagnosticFunctions
    import numpy as np
    import matplotlib.pyplot as plt
    import pprint

    group = np.random.rand(15,100)
    batch = np.array([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2])
    variance_ratio = DiagnosticFunctions.Variance_Ratios(group, batch)
    pprint.pprint(variance_ratio)

test_variance_ratio_multiple_batches()

def test_variance_ratio_plot():
    from DiagnoseHarmonization import DiagnosticFunctions
    from DiagnoseHarmonization import PlotDiagnosticResults
    import numpy as np
    import matplotlib.pyplot as plt
    import pprint

    group = np.random.rand(600,40)
    # 100 samples, 40 features
    batch = np.array([0]*200 + [1]*200 + [2]*200)
    # 3 batches
  
    variance_ratio = DiagnosticFunctions.Variance_Ratios(group, batch)
    pprint.pprint(variance_ratio)
    labels = [f"Batch {b1} vs Batch {b2}" for (b1,b2) in variance_ratio.keys()]

    ratio_array = np.array(list(variance_ratio.values()))
    print(np.shape(ratio_array))
    PlotDiagnosticResults.variance_ratio_plot(ratio_array, labels)
    print("Plotted Variance Ratio successfully.")
    plt.close("all")
