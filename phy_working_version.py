# Physics working version
# regression_trainer_IMRAN_v20.py
# --------------------------------------------------
# Baselines + Physics-guided XGBoost with AUTO-TUNING
# for Cij / rho prediction.
#
# - DecisionTree      : simple baseline
# - RandomForest      : tree ensemble baseline
# - XGBoost (baseline): raw features, fixed hyperparams
# - PhysicsXGB        : raw + physics features, hyperparameters
#                       tuned per dataset using Optuna (R² maximization)
#
# NO custom loss, NO sample weights, NO residual targets.
# --------------------------------------------------

import os
import time
import re
from math import sqrt

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

# Optuna for hyperparameter tuning
try:
    import optuna
except ImportError:
    raise ImportError("Optuna is required. Install with: pip install optuna")

# ============================================================
#  BASE XGBOOST PARAMS (for baseline XGBoost only)
# ============================================================

XGB_BASE_PARAMS = dict(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.3,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

# ============================================================
#  PHYSICS FEATURE BUILDERS
# ============================================================

def physics_features_cij_sci(X: pd.DataFrame) -> pd.DataFrame:
    """
    Physics-inspired features for Cij-SCI datasets.
    We NEVER remove original features, only add new ones with prefix 'phy_'.

    Expected columns include (from your screenshots):
      ['alpha_i', 'alpha_j', 'D_i', 'D_j',
       'D_acc_j', 'D_acc_i_pre', 'D_acc_ij',
       'D_acc_i_span', 'D_acc_j_span',
       'effective_length_i', 'effective_length_j',
       'length_i', 'length_j', 'Df']
    """
    df = X.copy()
    feat = pd.DataFrame(index=df.index)
    eps = 1e-9

    # ---- Symmetric span pair features for *_i / *_j ----
    pair_map = {}
    for col in df.columns:
        mi = re.match(r"(.+)_i(.*)$", col)
        mj = re.match(r"(.+)_j(.*)$", col)
        if mi:
            root, tail = mi.groups()
            pair_map.setdefault((root, tail), {})["i"] = col
        if mj:
            root, tail = mj.groups()
            pair_map.setdefault((root, tail), {})["j"] = col

    for (root, tail), sides in pair_map.items():
        if "i" in sides and "j" in sides:
            ci = sides["i"]
            cj = sides["j"]
            xi = df[ci]
            xj = df[cj]
            base = f"phy_{root}{tail}"
            feat[f"{base}_mean"] = 0.5 * (xi + xj)
            feat[f"{base}_absdiff"] = (xi - xj).abs()
            feat[f"{base}_prod"] = xi * xj

    # ---- Dispersion walk-off & mismatch (if available) ----
    if {"D_acc_i_span", "D_acc_j_span", "Df"}.issubset(df.columns):
        di = df["D_acc_i_span"]
        dj = df["D_acc_j_span"]
        dfreq = df["Df"].abs()
        diff = di - dj
        feat["phy_walkoff_abs"] = diff.abs()
        feat["phy_walkoff_weighted"] = diff.abs() * (dfreq ** 2)

    # ---- Length relations ----
    if {"length_i", "length_j"}.issubset(df.columns):
        li = df["length_i"]
        lj = df["length_j"]
        feat["phy_length_mean"] = 0.5 * (li + lj)
        feat["phy_length_absdiff"] = (li - lj).abs()

    # ---- Effective length relations ----
    if {"effective_length_i", "effective_length_j"}.issubset(df.columns):
        ei = df["effective_length_i"]
        ej = df["effective_length_j"]
        feat["phy_effL_mean"] = 0.5 * (ei + ej)
        feat["phy_effL_absdiff"] = (ei - ej).abs()
        feat["phy_effL_prod"] = ei * ej

    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat


def physics_features_cij_xci(X: pd.DataFrame) -> pd.DataFrame:
    """
    Physics-inspired features for Cij-XCI datasets.
    We rely more on power (pd_pp, pd_pr) and dispersion.
    """
    df = X.copy()
    feat = pd.DataFrame(index=df.index)
    eps = 1e-9

    # ---- power interactions ----
    if {"pd_pp", "pd_pr"}.issubset(df.columns):
        pp = df["pd_pp"]
        pr = df["pd_pr"]
        feat["phy_power_prod"] = pp * pr
        feat["phy_power_sum"] = pp + pr

    # ---- effective length product ----
    if {"effective_length_i", "effective_length_j"}.issubset(df.columns):
        ei = df["effective_length_i"]
        ej = df["effective_length_j"]
        feat["phy_effL_prod"] = ei * ej
        feat["phy_effL_mean"] = 0.5 * (ei + ej)

    # ---- dispersion walkoff ----
    if {"D_acc_i_span", "D_acc_j_span", "Df"}.issubset(df.columns):
        di = df["D_acc_i_span"]
        dj = df["D_acc_j_span"]
        dfreq = df["Df"].abs()
        diff = di - dj
        feat["phy_walkoff_abs"] = diff.abs()
        feat["phy_walkoff_weighted"] = diff.abs() * (dfreq ** 2)

    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat


def physics_features_rho(X: pd.DataFrame) -> pd.DataFrame:
    """
    Physics features for rho (SCI + XCI).
    Expected columns include:
      ['alpha','D','D_acc','D_acc_pre','D_acc_span',
       'length','effective_length','gamma','Rs','Df']
    """
    df = X.copy()
    feat = pd.DataFrame(index=df.index)
    eps = 1e-9

    # Nonlinear term gamma * effective_length
    if {"gamma", "effective_length"}.issubset(df.columns):
        g = df["gamma"]
        le = df["effective_length"]
        feat["phy_gamma_effL"] = g * le

    # Dispersion per length
    if {"D_acc", "length"}.issubset(df.columns):
        Dacc = df["D_acc"]
        L = df["length"]
        feat["phy_Dacc_perL"] = Dacc / (L + eps)
        feat["phy_Dacc_abs"] = Dacc.abs()

    # Loss * length
    if {"alpha", "length"}.issubset(df.columns):
        a = df["alpha"]
        L = df["length"]
        feat["phy_alphaL"] = a * L

    # Spectral load term Rs * |Df|
    if {"Rs", "Df"}.issubset(df.columns):
        Rs = df["Rs"]
        Df = df["Df"].abs()
        feat["phy_RsDf"] = Rs * Df

    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat


# ============================================================
#  PHYSICS-GUIDED XGB (FEATURES ONLY, SIMPLE LOSS)
# ============================================================

class PhysicsXGBRegressor:
    """
    Same loss as XGBoost (MSE).
    Difference: uses extra physics features with prefix 'phy_' added on top of raw X.
    Hyperparameters are passed in via **xgb_kwargs.
    """

    def __init__(self, target_type: str, dataset_name: str, **xgb_kwargs):
        assert target_type in ("Cij", "rho")
        self.target_type = target_type
        self.dataset_name = dataset_name.lower()

        # Start from a reasonable base, allow overrides from tuner
        params = dict(XGB_BASE_PARAMS)
        params.update(xgb_kwargs)
        self.model = XGBRegressor(**params)

    def _physics_block(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.target_type == "rho":
            return physics_features_rho(X)
        # Cij:
        if "sci" in self.dataset_name:
            return physics_features_cij_sci(X)
        else:
            return physics_features_cij_xci(X)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        phys = self._physics_block(X_df)
        X_enriched = pd.concat(
            [X_df.reset_index(drop=True), phys.reset_index(drop=True)],
            axis=1,
        )
        return X_enriched

    def fit(self, X, y):
        X_enriched = self._transform(pd.DataFrame(X))
        self.model.fit(X_enriched, y)
        return self

    def predict(self, X):
        X_enriched = self._transform(pd.DataFrame(X))
        return self.model.predict(X_enriched)


# ============================================================
#  OPTUNA TUNER FOR PhysicsXGB (ONLY)
# ============================================================

def tune_physics_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_type: str,
    dataset_name: str,
    n_trials: int = 30,
):
    """
    Tune PhysicsXGB hyperparameters ONLY, per dataset, using Optuna.
    Objective: maximize validation R².
    Uses a single fixed train/validation split of the training set.
    """
    # Fixed inner split (train/val) from train set only
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    def objective(trial: optuna.Trial):
        # Hyperparameter search space (moderately wide)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1e-1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        }

        model = PhysicsXGBRegressor(
            target_type=target_type,
            dataset_name=dataset_name,
            **params,
        )
        model.fit(X_tr, y_tr)
        preds_val = model.predict(X_val)
        r2 = r2_score(y_val, preds_val)
        return r2  # we will maximize R²

    print(f"\n🔍 Tuning PhysicsXGB for dataset='{dataset_name}', target='{target_type}'")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"✨ Best R² (val): {study.best_value:.6f}")
    print(f"✨ Best params: {study.best_params}")

    best_params = study.best_params
    return best_params, study.best_value


# ============================================================
#  TRAIN & EVAL
# ============================================================

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, preds)

    return mse, rmse, r2, preds, train_time


# ============================================================
#  MAIN
# ============================================================

def main():
    # ⚠ Update this path for your environment.
    # For Colab, put the CSVs in /content and set: data_dir = "/content"
    data_dir = r"C:\Users\dipto\Downloads\Projects\Imran ICMLCN Paper 2"
    results_dir = "results_regression_physics_v20"
    os.makedirs(results_dir, exist_ok=True)

    datasets = {
        "cijs_sci": ("dataset_cijs_sci_train.csv", "dataset_cijs_sci_test.csv"),
        "cijs_xci": ("dataset_cijs_xci_train.csv", "dataset_cijs_xci_test.csv"),
        "rho_sci": ("dataset_rho_sci_train.csv", "dataset_rho_sci_test.csv"),
        "rho_xci": ("dataset_rho_xci_train.csv", "dataset_rho_xci_test.csv"),
    }

    summary_records = []

    # Number of Optuna trials for PhysicsXGB (Option B: rich tuning)
    N_TRIALS = 30

    for name, (train_file, test_file) in datasets.items():
        print(f"\n📘 Processing dataset: {name}")

        train_df = pd.read_csv(os.path.join(data_dir, train_file))
        test_df = pd.read_csv(os.path.join(data_dir, test_file))

        # Detect target
        if "Cij" in train_df.columns:
            target = "Cij"
        elif "rho" in train_df.columns:
            target = "rho"
        else:
            raise ValueError("Target must be 'Cij' or 'rho'")

        X_train = train_df.drop(columns=[target])
        y_train = train_df[target]
        X_test = test_df.drop(columns=[target])
        y_test = test_df[target]

        print(f"  → Target: {target} | Train: {X_train.shape} | Test: {X_test.shape}")

        # -------------------------------------------------------
        # 1) BASELINES (no tuning here to highlight Phy-XGB gain)
        # -------------------------------------------------------
        baseline_models = {
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "RandomForest": RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                random_state=42,
            ),
            "XGBoost": XGBRegressor(**XGB_BASE_PARAMS),
        }

        # -------------------------------------------------------
        # 2) TUNE PhysicsXGB ON THIS DATASET (train only)
        # -------------------------------------------------------
        best_phy_params, best_val_r2 = tune_physics_xgb(
            X_train, y_train, target_type=target, dataset_name=name, n_trials=N_TRIALS
        )

        physics_model = PhysicsXGBRegressor(
            target_type=target,
            dataset_name=name,
            **best_phy_params,
        )

        models = {
            **baseline_models,
            "PhysicsXGB": physics_model,
        }

        result_path = os.path.join(results_dir, name)
        os.makedirs(result_path, exist_ok=True)

        # -------------------------------------------------------
        # 3) FINAL TRAINING ON FULL TRAIN + TEST EVAL
        # -------------------------------------------------------
        for model_name, model in models.items():
            print(f"  ⚙️ Training {model_name}...")
            mse, rmse, r2, preds, train_time = train_and_evaluate(
                model, X_train, y_train, X_test, y_test
            )

            pd.DataFrame({"True": y_test, "Predicted": preds}).to_csv(
                os.path.join(result_path, f"{model_name}_predictions.csv"),
                index=False,
            )

            print(
                f"     → {model_name}: MSE={mse:.6f}, RMSE={rmse:.6f}, "
                f"R²={r2:.4f}, Time={train_time:.2f}s"
            )

            summary_records.append(
                {
                    "Dataset": name,
                    "Model": model_name,
                    "Target": target,
                    "MSE": mse,
                    "RMSE": rmse,
                    "R2": r2,
                    "Training_Time(s)": train_time,
                }
            )

        # Also store best validation R² from tuning for PhysicsXGB
        summary_records.append(
            {
                "Dataset": name,
                "Model": "PhysicsXGB (tuning_val)",
                "Target": target,
                "MSE": np.nan,
                "RMSE": np.nan,
                "R2": best_val_r2,
                "Training_Time(s)": np.nan,
            }
        )

    pd.DataFrame(summary_records).to_csv(
        os.path.join(results_dir, "all_results_summary_v20.csv"),
        index=False,
    )

    print("\n🎉 All models (including auto-tuned PhysicsXGB) evaluated successfully!")
    print(f"📁 Results stored in: {results_dir}")


if __name__ == "__main__":
    main()
