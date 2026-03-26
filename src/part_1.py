"""
part1.py

Train a leakage-free regression model for target01 using dataset_58.
The pipeline performs median imputation, controlled numeric interaction
feature engineering, and LightGBM regression with log-transformed targets.
Model performance is evaluated using 5-fold cross-validation with MAE,
RMSE, and R² metrics. After validation, the final model is trained on the
full dataset and used to generate predictions for the evaluation set.

Outputs:
  • Prints training and validation MAE, RMSE, and R² for each fold.
  • Prints mean validation MAE, RMSE, and R² across all folds.
  • Saves predictions for evaluation data as "EVAL_target01_58.csv".

Usage:
    python part1.py

References:    
    Regression problem solving code reference:
        https://github.com/ankishb/ml-projects/tree/master/jp-morgan

    Hybrid evaluation pipeline: function made with the help of ChatGPT 
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

from lightgbm import LGBMRegressor


# =====================================================
# Configuration
# =====================================================
class Config:
    """
    Configuration class holding all constants and paths
    used throughout the training and prediction pipeline.
    """
    RANDOM_STATE = 42
    N_SPLITS = 5
    TOP_K_INTERACTIONS = 20

    DATA_PATH = "problem_58/dataset_58.csv"
    TARGET_PATH = "problem_58/target_58.csv"
    TARGET_COL = "target01"

    EVAL_DATA_PATH = "problem_58/EVAL_58.csv"
    OUTPUT_PATH = "problem_58/EVAL_target01_58.csv"

# Feature Engineering
class InteractionFeatureGenerator:
    """
    Utility class for generating numeric × numeric
    interaction features in a controlled manner.
    """

    @staticmethod
    def fit_transform(df: pd.DataFrame, top_k: int):
        """
        Selects the top-k highest variance features from the
        training data and generates pairwise multiplicative
        interaction features.

        Parameters
        ----------
        df : pd.DataFrame
            Training feature matrix after imputation.
        top_k : int
            Number of high-variance features to use for interactions.

        Returns
        -------
        df_out : pd.DataFrame
            Feature matrix with interaction features added.
        selected_features : list
            List of base feature names used to create interactions.
        """
        variances = df.var(axis=0).sort_values(ascending=False)
        selected_features = variances.head(top_k).index.tolist()

        df_out = df.copy()
        for i in range(len(selected_features)):
            for j in range(i + 1, len(selected_features)):
                f1, f2 = selected_features[i], selected_features[j]
                df_out[f"{f1}_x_{f2}"] = df_out[f1] * df_out[f2]

        return df_out, selected_features

    @staticmethod
    def transform(df: pd.DataFrame, selected_features: list):
        """
        Applies the same interaction feature construction
        to validation or evaluation data.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix after imputation.
        selected_features : list
            Base features selected during training.

        Returns
        -------
        df_out : pd.DataFrame
            Feature matrix with interaction features added.
        """
        df_out = df.copy()
        for i in range(len(selected_features)):
            for j in range(i + 1, len(selected_features)):
                f1, f2 = selected_features[i], selected_features[j]
                df_out[f"{f1}_x_{f2}"] = df_out[f1] * df_out[f2]

        return df_out

# Model Trainer
class Target01Trainer:
    """
    Encapsulates model training, cross-validation,
    and prediction logic for target01.
    """

    def __init__(self, config: Config):
        """
        Initializes the model, imputer, and configuration.

        Parameters
        ----------
        config : Config
            Configuration object containing hyperparameters and paths.
        """
        self.cfg = config
        self.model = LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=30,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
        self.imputer = SimpleImputer(strategy="median")
        self.selected_features = None

    def cross_validate(self, X: pd.DataFrame, y: pd.Series):
        """
        Performs leakage-free K-Fold cross-validation.
        Prints training and validation metrics for each fold.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target values.
        """
        kf = KFold(
            n_splits=self.cfg.N_SPLITS,
            shuffle=True,
            random_state=self.cfg.RANDOM_STATE
        )

        mae_scores, rmse_scores, r2_scores = [], [], []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Imputation
            X_train = pd.DataFrame(
                self.imputer.fit_transform(X_train_raw),
                columns=X.columns,
                index=X_train_raw.index
            )
            X_val = pd.DataFrame(
                self.imputer.transform(X_val_raw),
                columns=X.columns,
                index=X_val_raw.index
            )

            # Feature engineering
            X_train_fe, selected = InteractionFeatureGenerator.fit_transform(
                X_train, self.cfg.TOP_K_INTERACTIONS
            )
            X_val_fe = InteractionFeatureGenerator.transform(X_val, selected)

            # Train on log-transformed target
            self.model.fit(X_train_fe, np.log1p(y_train))

            # Predictions (real scale)
            train_pred = np.expm1(self.model.predict(X_train_fe))
            val_pred = np.expm1(self.model.predict(X_val_fe))

            # Metrics
            train_mae = mean_absolute_error(y_train, train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            train_r2 = r2_score(y_train, train_pred)

            val_mae = mean_absolute_error(y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_r2 = r2_score(y_val, val_pred)

            mae_scores.append(val_mae)
            rmse_scores.append(val_rmse)
            r2_scores.append(val_r2)

            print(f"\nFold {fold}")
            print(f"TRAIN → MAE : {train_mae:.4f}")
            print(f"TRAIN → RMSE: {train_rmse:.4f}")
            print(f"TRAIN → R²  : {train_r2:.4f}")
            print(f"VAL   → MAE : {val_mae:.4f}")
            print(f"VAL   → RMSE: {val_rmse:.4f}")
            print(f"VAL   → R²  : {val_r2:.4f}")

        print("\n==============================")
        print("FINAL CV RESULTS (VALIDATION)")
        print(f"Mean MAE  : {np.mean(mae_scores):.4f}")
        print(f"Mean RMSE : {np.mean(rmse_scores):.4f}")
        print(f"Mean R²   : {np.mean(r2_scores):.4f}")
        print("==============================")

    def train_full(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the final model on the full dataset
        after cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            Full feature matrix.
        y : pd.Series
            Full target vector.
        """
        X_imp = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        X_fe, self.selected_features = InteractionFeatureGenerator.fit_transform(
            X_imp, self.cfg.TOP_K_INTERACTIONS
        )

        self.model.fit(X_fe, np.log1p(y))
        print("\n Final model trained on full dataset")

    def predict_eval(self, X_eval: pd.DataFrame):
        """
        Generates predictions for evaluation data.

        Parameters
        ----------
        X_eval : pd.DataFrame
            Evaluation feature matrix.

        Returns
        -------
        np.ndarray
            Predicted target01 values (original scale).
        """
        X_eval_imp = pd.DataFrame(
            self.imputer.transform(X_eval),
            columns=X_eval.columns,
            index=X_eval.index
        )

        X_eval_fe = InteractionFeatureGenerator.transform(
            X_eval_imp, self.selected_features
        )

        return np.expm1(self.model.predict(X_eval_fe))

# Main Execution
def main():
    """
    Main entry point for training, validation,
    and evaluation prediction.
    """
    cfg = Config()
    np.random.seed(cfg.RANDOM_STATE)

    X = pd.read_csv(cfg.DATA_PATH).replace([np.inf, -np.inf], np.nan)
    y = pd.read_csv(cfg.TARGET_PATH)[cfg.TARGET_COL]

    print("Loaded:")
    print("X:", X.shape, "| y:", y.shape)

    trainer = Target01Trainer(cfg)

    trainer.cross_validate(X, y)
    trainer.train_full(X, y)

    X_eval = pd.read_csv(cfg.EVAL_DATA_PATH).replace([np.inf, -np.inf], np.nan)
    preds = trainer.predict_eval(X_eval)

    pd.DataFrame({"target01": preds}).to_csv(cfg.OUTPUT_PATH, index=False)
    print(f"\n Predictions saved to {cfg.OUTPUT_PATH}")


if __name__ == "__main__":
    main()
