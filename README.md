# DiagnoseHarmonization version 0.0.1

This library can be divided into two parts:

## DiagnosticReport

Take in arguments for data, covariates and batch and returns a pandas dataframe with the results of a set of statistical tests as well as producing a log file which contains summary statistics and advice for best approaches when harmonizing multisite or batched data.

## PlotDiagnostics

Use the output from DiagnoseHarmonization to produce research quality plots of various statistical tests of batch
