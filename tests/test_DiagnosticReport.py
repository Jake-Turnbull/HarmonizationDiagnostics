import os
import numpy as np
from pathlib import Path
import pytest
import time
import webbrowser

# Adjust import to match your package layout
# from DiagnoseHarmonization.DiagnosticReport import DiagnosticReport
# If DiagnosticReport is defined in a module file DiagnosticReport.py inside DiagnoseHarmonization package:
from DiagnoseHarmonization import DiagnosticReport

save_dir = "/Users/jacob.turnbull/VS_code_projects/"

def test_full_pipeline_generates_report(tmp_path = save_dir):
    """
    Run the full DiagnosticReport pipeline once and produce a single HTML report.
    - Writes report to a temporary directory (tmp_path / "diagnostic_full_run")
    - Asserts that a timestamped DiagnosticReport_*.html file was created
    - Prints the path to the report for manual viewing
    - Optionally opens the report automatically when OPEN_REPORT=1
    """
    # -------------------------
    # Prepare synthetic data
    # -------------------------
    np.random.seed(27)
    n_samples = 80
    n_features = 100

    data = np.random.randn(n_samples, n_features)
    covariate_cat = np.random.randint(0, 1, size=n_samples)    # categorical
    print( covariate_cat)
    # Mean center the categorical covariate, testing this as divide by zero errors in PCA correlations otherwise
    covariate_cat = covariate_cat - np.mean(covariate_cat) # Try mean centering as correlations with PCA not working 27/10/2025
    batch = np.array(["Siemens"] * 20 + ["Philips"] * 20  + ["GE"] * 20  + ["Magnetom"] * 20 )
    # Construct mixed effects model to add some batch and covariate effects
    covariate_cont = np.random.rand(n_samples)                # continuous
    covariates = np.column_stack((covariate_cat, covariate_cont))
    variable_names = ['Sex', 'Age']

    for i in range(n_samples):
        if batch[i] == "Siemens":
            data[i, :] += 1.0
        elif batch[i] == "Philips":
            data[i, :] -= 2.0
        elif batch[i] == "GE":
            data[i, :] += 3.0
        elif batch[i] == "Magnetom":
            data[i, :] += 4.0

    # Simulate covariate effect of age and sex (when age increases, feature values decrease) 
    # (when sex = 0/1 (female/male), feature values decrease/increase to simulate volume differences)
    for i in range(n_samples):
        data[i, :] += covariates[i, 1] * -0.5 * np.random.rand()  # age effect
        data[i, :] += 0.5 * np.random.rand()
    # -------------------------
    # Where to save the report
    # -------------------------
    out_dir = tmp_path + "diagnostic_full_run"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Run the DiagnosticReport
    # -------------------------
    # If your DiagnosticReport is exported as a function in the module,
    # adjust the call accordingly. Here we expect a function named DiagnosticReport.
    # If instead the function lives at DiagnoseHarmonization.DiagnosticReport.DiagnosticReport,
    # change the call accordingly.
    try:
        # call signature:
        # DiagnosticReport(data, batch, covariates=None, variable_names=None,
        #                  save_dir=None, SaveArtifacts=False, rep=None, show=False)
        DiagnosticReport.DiagnosticReport(
            data=data,
                batch=batch,
                    covariates=covariates,
                        covariate_names=variable_names,
                            save_dir=str(out_dir),
                                SaveArtifacts=False,
                                    rep=None,
                                        show=False
        )
    except Exception as e:
        # If the pipeline raises, fail the test but show the exception for debugging
        pytest.fail(f"DiagnosticReport raised an exception: {e}")

    # -------------------------
    # Find the generated report
    # -------------------------
    reports = sorted(out_dir.glob("DiagnosticReport_*.html"))
    assert len(reports) > 0, f"No DiagnosticReport_*.html file found in {out_dir}"

    # pick the most recent report
    report_path = reports[-1]

    # Basic sanity checks
    assert report_path.exists() and report_path.stat().st_size > 100, "Report file is missing or unexpectedly small."

    # Print the full path so the tester can open it manually; -s flag required to see this in pytest output
    print("\n==== Diagnostic report generated ====")
    print(f"Report path: {report_path}")
    print("Open this file in your browser to view the report.")
    print("====================================\n")

    # Success
    assert True