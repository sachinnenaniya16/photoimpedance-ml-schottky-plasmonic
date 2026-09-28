# Wavelength-Resolved Photo-Impedance and Leakage-Aware Machine Learning Reveal a Schottky-to-Plasmonic Mechanism Transition in Ag/Au/g-C₃N₄ Photocathodes for Hydrogen Evolution

This repository contains the complete data-analysis and machine-learning pipeline supporting the manuscript above. It links wavelength-resolved photo-electrochemical impedance spectroscopy (EIS) of plasmonic metal (Ag, Au) / g-C₃N₄ photocathodes to their photocatalytic hydrogen-evolution performance, and uses interpretable machine learning to identify which electrochemical descriptors govern that performance across illumination conditions.

The dataset comprises 7 catalysts (bare g-C₃N₄ and Ag/Au at three loadings each) measured under 4 illumination conditions (dark, 410 nm, 440 nm, full spectrum) at 3 applied voltages — 84 rows in total, each carrying EIS-derived descriptors (charge-transfer resistance, relaxation time, Nyquist-derived quantities), linear-sweep voltammetry descriptors (Tafel slope, onset potential), chronoamperometry stability metrics, and structural/optical characterization (TEM, XRD, steady-state and time-resolved PL, UV-Vis).

## Repository structure

```
01_data_preparation.ipynb      Assembly and cleaning of the 84-row master table from raw instrument files
02_correlation_analysis.ipynb  Pairwise descriptor-target correlation analysis
03_PCA.ipynb                   Principal component analysis of the descriptor space
04_RF_SHAP_PDP.ipynb           Random Forest models with SHAP importance and partial dependence
05_GPR_models.ipynb            Gaussian Process Regression and linear (LASSO) models per target
06_prediction_benchmark.ipynb  Hyperparameter-tuned RF, GPR, SVR, GB, and ElasticNet benchmark
SI_Raw_EIS_Figures.ipynb       Supplementary raw Nyquist plots for all seven catalysts
utils.py                       Shared constants and functions imported by every notebook
Dataset/                       Raw instrument files (EIS .DTA, LSV, TEM, XRD, PL, UV-Vis)
Results/                       Generated master table, figures, and saved model artifacts
Publication/                   Manuscript and ESI source files
```

`utils.py` is the single source of truth for the feature list, target definitions, plotting conventions, and the cross-validation/scoring routines. Every notebook imports from it rather than redefining these locally, so results are consistent by construction across the whole project.

## Methodology

**Grouped validation.** The master table has 84 rows: 7 catalysts × 4 illumination conditions × 3 voltages. The three voltage rows for a given catalyst × condition are not independent measurements for most targets, so all cross-validation is performed at the level of the 28 catalyst × condition groups (Leave-One-Group-Out), never at the level of individual rows.

**Target-specific feature sets.** A small number of EIS-derived columns are constructed as direct algebraic functions of one particular target (an identity, a group mean, a voltage-derivative, or a ratio). Each such column is a valid predictor for every target except the one it is derived from, so it is excluded only from that target's feature set (`DERIVED_FROM_TARGET` in `utils.py`):

| Target | Excluded from its own feature set |
|---|---|
| Relaxation time (`tau_s`) | `peak_freq_Hz` |
| LSPR resistance drop (`delta_Rct`) | `delta_Rct_mean`, `dDeltaRct_dV` |
| Charge-transfer resistance (`Rct_ohm`) | `Rct_ratio`, `Rct_ratio_mean` |

**Model selection.** Five targets are modeled: photocurrent at −0.7 V, charge-transfer resistance, LSPR-induced resistance drop, relaxation time, and Tafel slope. Notebook 05 establishes the primary (reported) model per target; Notebook 06 benchmarks tuned Random Forest, Gaussian Process, Support Vector, Gradient Boosting, and ElasticNet regressors against those reported models under identical grouped cross-validation, with 95% confidence intervals obtained by group-level bootstrap (2000 resamples).

## Key results

Grouped-LOOCV R² (with 95% bootstrap CI) for the reported model of each target:

| Target | Model | R² | 95% CI |
|---|---|---|---|
| Photocurrent J (−0.7 V) | LASSO (α = 0.05) | 0.533 | [0.269, 0.766] |
| Charge-transfer resistance | GPR | 0.909 | [0.877, 0.947] |
| LSPR resistance drop | LASSO (α = 0.01) | 0.813 | [0.658, 0.895] |
| Relaxation time | GPR | 0.860 | [0.827, 0.898] |
| Tafel slope | RF | −0.552 | [−1.003, −0.268] |

See `06_prediction_benchmark.ipynb` for the full hyperparameter-tuned comparison against these reported values.

## Reproducing the analysis

Requirements: Python ≥ 3.10 with

```
numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
shap
jupyter
nbformat
```

Install with:

```
pip install numpy pandas scikit-learn scipy matplotlib seaborn shap jupyter nbformat
```

Run the notebooks in numerical order from the repository root (01 → 06); `SI_Raw_EIS_Figures.ipynb` is independent and can be run at any point. Each notebook reads from `Dataset/` and/or `Results/MASTER_TABLE_n84.csv` and writes its outputs back to `Results/`.

## Author

Sachin Nenaniya

## Citation

If you use this code or dataset, please cite the associated manuscript (citation details to be added upon publication).
