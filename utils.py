"""
Shared definitions and helper functions for the descriptor-activity analysis
of plasmonic Ag/Au g-C3N4 photocatalysts.

Every notebook in this project imports from here instead of redefining the
feature list, the color/order conventions, or the cross-validation and
scoring routines locally. Centralizing them keeps every notebook's numbers
consistent by construction.

Methodology notes
------------------
Two conventions are used throughout and are implemented here once:

1. Grouped validation. The master table has 84 rows: 7 catalysts x 4
   illumination conditions x 3 voltages. The three voltage rows for a given
   catalyst x condition are not independent measurements of most targets
   (e.g. photocurrent and the electrochemical descriptors used to predict
   it), so cross-validation is always done at the level of the 28
   catalyst x condition groups (LeaveOneGroupOut), never at the level of
   individual rows.

2. Target-specific feature sets. A few EIS-derived columns are constructed
   as direct algebraic functions of one particular target (an identity,
   a group mean, a derivative, or a ratio). Each such column is a valid
   predictor for every target except the one it is derived from, so it is
   excluded only from that one target's feature set. See
   DERIVED_FROM_TARGET and feature_columns() below.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

MASTER_TABLE = "Results/MASTER_TABLE_n84.csv"
SEED = 42

# ---------------------------------------------------------------------------
# Features and targets
# ---------------------------------------------------------------------------

FEATURES = [
    "metal_num", "loading_num", "condition_num", "voltage",
    "Rs_ohm", "Rct_ohm", "peak_freq_Hz", "tau_s", "phase_min", "Cdl_est_F",
    "Rct_dark", "delta_Rct", "Rct_ratio", "delta_Rs",
    "dDeltaRct_dV", "Rct_ratio_mean", "delta_Rct_mean",
    "Vfb_V", "Eg_eV", "E_onset_V", "|tafel|_mV_dec",
    "CA_dJ_mean_mA", "CA_dJ_std_mA", "CA_spike_mA", "CA_stability",
    "TEM_mean_nm", "TEM_std_nm", "XRD_gCN_tau_nm", "XRD_gCN_layers",
    "XRD_metal_tau_nm", "SSPL_quenching_ratio",
    "TRPL_tau_A_ns", "TRPL_tau_int_ns", "UV_Eg_eV",
]

TARGETS = {
    "J_m0p7":         "Photocurrent J (-0.7 V)",
    "Rct_ohm":        "Charge transfer resistance",
    "delta_Rct":      "LSPR resistance drop",
    "tau_s":          "Relaxation time",
    "|tafel|_mV_dec": "Tafel slope",
}

# Observed span of each target, used to compute NRMSE% (RMSE normalized by
# the range of the measured values, for easy comparison across targets).
TARGET_RANGE = {
    "J_m0p7": 2.33,
    "Rct_ohm": 12800,
    "delta_Rct": 9000,
    "tau_s": 0.155,
    "|tafel|_mV_dec": 264,
}

# Descriptors that are constructed as a direct algebraic function of one
# target (identity, group mean, voltage-derivative, or ratio) and are
# therefore excluded only from that target's own feature set.
DERIVED_FROM_TARGET = {
    "tau_s":     ["peak_freq_Hz"],                    # tau = 1 / (2*pi*f_peak)
    "delta_Rct": ["delta_Rct_mean", "dDeltaRct_dV"],  # group mean / d/dV of delta_Rct
    "Rct_ohm":   ["Rct_ratio", "Rct_ratio_mean"],     # Rct_ohm / Rct_dark and its group mean
}

# ---------------------------------------------------------------------------
# Plot conventions
# ---------------------------------------------------------------------------

SAMPLE_ORDER = ["C3N4", "Ag0.25", "Ag0.5", "Ag1", "Au0.25", "Au0.5", "Au1"]
SAMPLE_COLORS = {
    "C3N4":   "#555555",
    "Ag0.25": "#9B59B6", "Ag0.5": "#2471A3", "Ag1": "#1ABC9C",
    "Au0.25": "#E67E22", "Au0.5": "#E74C3C", "Au1": "#F1C40F",
}
SAMPLE_MARKERS = {
    "C3N4": "o",
    "Ag0.25": "s", "Ag0.5": "s", "Ag1": "s",
    "Au0.25": "^", "Au0.5": "^", "Au1": "^",
}
CONDITION_ORDER = ["Dark", "410 nm", "440 nm", "Full spectrum"]
CONDITION_COLORS = {
    "Dark": "#555555", "410 nm": "#9B59B6",
    "440 nm": "#2471A3", "Full spectrum": "#27AE60",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_master(path=MASTER_TABLE):
    """Load the master table and attach a group id (catalyst x condition,
    28 groups of 3 voltage-rows each) used for grouped cross-validation."""
    df = pd.read_csv(path)
    df["group_id"] = df["sample"].astype(str) + "_" + df["condition"].astype(str)
    return df


def feature_columns(target, features=FEATURES):
    """Feature list for a given target: every descriptor except the target
    itself and the descriptors derived from it (see DERIVED_FROM_TARGET)."""
    drop = set(DERIVED_FROM_TARGET.get(target, [])) | {target}
    return [f for f in features if f not in drop]


def prepare(df, target, features=FEATURES):
    """Return X, y, group labels and the feature names used for a target."""
    cols = feature_columns(target, features)
    d = df[cols + [target, "group_id"]].dropna().reset_index(drop=True)
    return d[cols].values, d[target].values, d["group_id"].values, cols

# ---------------------------------------------------------------------------
# Cross-validation, scoring, bootstrap
# ---------------------------------------------------------------------------

def grouped_loo_predict(factory, X, y, groups, scale=False):
    """Out-of-fold predictions under LeaveOneGroupOut. `factory` returns a
    fresh unfitted model instance. Scaling, if requested, is fit inside
    each training fold only."""
    pred = np.empty(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        Xtr, Xte = X[tr], X[te]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        model = factory()
        model.fit(Xtr, y[tr])
        pred[te] = model.predict(Xte)
    return pred


def score(y, pred, target):
    """R2, RMSE, MAE and NRMSE% for a set of (grouped-CV) predictions."""
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    return {
        "r2": r2_score(y, pred),
        "rmse": rmse,
        "mae": mean_absolute_error(y, pred),
        "nrmse_pct": 100 * rmse / TARGET_RANGE[target],
    }


def bootstrap_r2_ci(y, pred, groups, n_boot=2000, seed=SEED):
    """95% CI for R2 by resampling whole catalyst x condition groups (not
    individual rows) with replacement, n_boot times."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    vals = []
    for _ in range(n_boot):
        chosen = rng.choice(ids, size=len(ids), replace=True)
        mask = np.isin(groups, chosen)
        if mask.sum() > 1 and np.unique(y[mask]).size > 1:
            vals.append(r2_score(y[mask], pred[mask]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
