import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("Simple simulator with persistent data (use Run Simulation first)")

# UI controls
number_batches = st.slider("Number of Batches", min_value=2, max_value=10, value=3, step=1)
samples_per_batch = st.slider("Samples per Batch", min_value=10, max_value=500, value=40, step=10)
feature_dim = st.slider("Feature Dimension", min_value=10, max_value=100, value=50, step=10)
covariates = st.multiselect("Select Covariates to Simulate",
                            options=["Age", "Sex", "Diagnosis"],
                            default=["Age"])
batch_additive_severity = st.slider("Batch Mean Effect Severity", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
batch_multiplicative_severity = st.slider("Batch Variance Effect Severity", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

# Run simulation and store results in session_state so they persist across reruns
if st.button("Run Simulation"):
    st.write("Generating Data...")
    total_samples = number_batches * samples_per_batch

    # Base data, randomly drawn from normal distribution around zero
    data = np.random.randn(total_samples, feature_dim)

    # Covariates
    covariate_data = pd.DataFrame(index=range(total_samples))
    if "Age" in covariates:
        covariate_data["Age"] = 20 + 60 * np.random.rand(total_samples) # Age range 20-80
    if "Sex" in covariates:
        covariate_data["Sex"] = np.random.randint(0, 2, size=total_samples)
    if "Diagnosis" in covariates:
        covariate_data["Diagnosis"] = np.random.randint(0, 5, size=total_samples)

    # Batch labels
    batch_labels = np.array([f"Batch_{i+1}" for i in range(number_batches) for _ in range(samples_per_batch)])

    # Introduce batch effects (vectorized where possible)
    scale_severity = np.linspace(1, batch_multiplicative_severity, number_batches)  # per-batch shapes
    add_severity = np.linspace(-batch_additive_severity, batch_additive_severity, number_batches)  # per-batch locs

    # Loop over batches but avoid inner-most redundant loops where possible
    for i in range(number_batches):
        start = i * samples_per_batch
        end = start + samples_per_batch

        # Apply additive and multiplicative effects, one specific scaling term per batch per feature (solumn)
        # Draw from inverse gamma for each feature in the batch, equivalent to delta * epsilon per feature where epsilon ~ N(0,1) and is the subject specific measurement noise
        data[start:end, :] = (data[start:end, :] * np.random.gamma(shape=2.0/scale_severity[i], scale=scale_severity[i], size=(samples_per_batch, feature_dim)))

        # Additive effect drawn from random normal distribution across features per batch, centered around the batch-specific loc
        data[start:end, :] += np.random.normal(loc=add_severity[i], scale=0.1, size=(samples_per_batch, feature_dim))


    # Covariate effects (vectorized), assume same distribution across batches, severity is fixed across features here for simplicity (this is an unrealistic assumption that almost never holds)
    if "Age" in covariates:
        age_centered = covariate_data["Age"].values - covariate_data["Age"].mean()
        data += (-0.05) * age_centered.reshape(-1, 1)
    if "Sex" in covariates:
        sex_centered = covariate_data["Sex"].values - covariate_data["Sex"].mean()
        data += (0.25) * sex_centered.reshape(-1, 1) 
    if "Diagnosis" in covariates:
        diag_centered = covariate_data["Diagnosis"].values - covariate_data["Diagnosis"].mean()
        data += (0.5) * diag_centered.reshape(-1, 1)

    # Save to session_state
    st.session_state["data"] = data
    st.session_state["batch_labels"] = batch_labels
    st.session_state["covariate_data"] = covariate_data
    st.session_state["samples_per_batch"] = samples_per_batch
    st.session_state["number_batches"] = number_batches
    st.session_state["feature_dim"] = feature_dim

    st.write("Data Generation Complete.")

# Feature slider (use session_state feature_dim if available)
max_feature = st.session_state.get("feature_dim", feature_dim) - 1
feature_to_plot = st.slider("Select Feature to Plot Histogram and Boxplots", min_value=0, max_value=max_feature, value=0, step=1)


# Plot only if data exists
if "data" not in st.session_state:
    st.warning("No data yet — please click **Run Simulation** to generate data before plotting.")
else:
    data = st.session_state["data"]
    number_batches = st.session_state["number_batches"]
    samples_per_batch = st.session_state["samples_per_batch"]

    fig, ax = plt.subplots()
    for i in range(number_batches):
        start = i * samples_per_batch
        end = start + samples_per_batch
        batch_data = data[start:end, feature_to_plot]
        ax.hist(batch_data, bins=30, alpha=0.5, label=f"Batch_{i+1}")
    ax.set_title(f"Histogram of Feature {feature_to_plot} by Batch")
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)
    fig2, ax   = plt.subplots()
    ax.boxplot([data[i*samples_per_batch:(i+1)*samples_per_batch, feature_to_plot] for i in range(number_batches)],
               labels=[f"Batch_{i+1}" for i in range(number_batches)])
    ax.set_title(f"Boxplot of Feature {feature_to_plot} by Batch")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Feature Value")
    st.pyplot(fig2)

# New button that runs CrossSectionalReport on the data and batches in session_state, keep name fixed and save to current directory so only one report exists at a time

if st.button("Generate Cross-Sectional Report"):
    if "data" not in st.session_state:
        st.warning("No data yet — please click **Run Simulation** to generate data before generating report.")
    else:
        from DiagnoseHarmonization import DiagnosticReport
        data = st.session_state["data"]
        batch_labels = st.session_state["batch_labels"]
        covariate_data = st.session_state["covariate_data"]
        st.write("Generating Cross-Sectional Report...")
        report = DiagnosticReport.CrossSectionalReport(data,
                                                        batch_labels,
                                                          covariate_data,
                                                            covariate_names=covariate_data.columns.tolist(),
                                                                save_dir=".",
                                                                    save_data=False,
                                                                    report_name="Simulator_Report",
                                                                        SaveArtifacts=False,
                                                                        rep=None,
                                                                            show=False
                                                          )
        report_filename = "Simulator_Report.html"
        st.write(f"Report generated and saved as {report_filename}. You can open this file in your web browser to view the report.")