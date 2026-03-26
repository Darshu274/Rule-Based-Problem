"""
part_2.py

Load dataset_58.csv and target_58.csv, identify a small subset of relevant features,
and extract simple, interpretable rules for predicting target02 using a shallow
decision tree. The learned rules are intended for deployment on an edge device
that supports only basic comparisons and numerical calculations.

The pipeline performs the following steps:
  • Load feature data and target02 values.
  • Analyze feature–target correlation to identify candidate features.
  • Train a shallow DecisionTreeRegressor offline to discover simple threshold-based rules.
  • Visualize and print the extracted rules for inspection.
  • Evaluate the rule-based model using RMSE and R² metrics.
  • Use the extracted rules as a deterministic, non-ML prediction mechanism.

Usage:
    python part_2.py

References:
    Decision Trees (Scikit-learn):
        https://scikit-learn.org/stable/modules/tree.html

    Rule-Based Modeling for Interpretability:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    Edge-Deployable Rule Systems:
        Quinlan, J. R. (1986). Induction of decision trees. Machine Learning.

    Correlation Analysis:
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corrwith.html

    Hybrid evaluation pipeline: function made with the help of ChatGPT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree


class RuleBasedModelExtractor:
    """
    Class for extracting simple rule-based models for target02
    using a shallow decision tree.
    """

    def __init__(self, dataset_path: str, target_path: str, random_state: int = 42):
        """
        Initialize the rule extractor.

        Parameters
        ----------
        dataset_path : str
            Path to dataset_58.csv
        target_path : str
            Path to target_58.csv
        random_state : int
            Random seed for reproducibility
        """
        self.dataset_path = dataset_path
        self.target_path = target_path
        self.random_state = random_state

        self.X = None
        self.y = None
        self.tree = None
        self.important_features = None

    def load_data(self) -> None:
        """
        Load feature matrix and target02 values.
        """
        self.X = pd.read_csv(self.dataset_path)
        target_df = pd.read_csv(self.target_path)
        self.y = target_df["target02"]

        print("Data loaded successfully.")
        print("X shape:", self.X.shape)
        print("target02 shape:", self.y.shape)

    def analyze_feature_correlation(self, top_k: int = 15) -> pd.Series:
        """
        Compute absolute correlation between features and target02.

        Parameters
        ----------
        top_k : int
            Number of top correlated features to display

        Returns
        -------
        pd.Series
            Sorted correlation values
        """
        corr = self.X.corrwith(self.y).abs().sort_values(ascending=False)
        print(f"\nTop {top_k} correlated features:")
        print(corr.head(top_k))
        return corr

    def select_features(self, features: list) -> None:
        """
        Select a subset of important features for rule extraction.

        Parameters
        ----------
        features : list
            List of feature names
        """
        self.important_features = features
        self.X = self.X[features].copy()
        print("\nSelected features:", features)

    def train_shallow_tree(self, max_depth: int = 3, min_samples_leaf: int = 50) -> None:
        """
        Train a shallow decision tree to extract simple rules.

        Parameters
        ----------
        max_depth : int
            Maximum depth of the decision tree
        min_samples_leaf : int
            Minimum samples required per leaf
        """
        self.tree = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state
        )
        self.tree.fit(self.X, self.y)
        print("\nShallow decision tree trained.")

    def print_extracted_rules(self) -> None:
        """
        Print the extracted decision rules in text format.
        """
        rules = export_text(self.tree, feature_names=self.important_features)
        print("\nExtracted Decision Rules:\n")
        print(rules)

    def visualize_tree(self) -> None:
        """
        Visualize the trained decision tree.
        """
        plt.figure(figsize=(20, 10))
        plot_tree(
            self.tree,
            feature_names=self.important_features,
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title("Shallow Decision Tree for Rule-Based Target02")
        plt.tight_layout()
        plt.show()

    def evaluate_model(self) -> dict:
        """
        Evaluate the rule-based model on the full dataset.

        Returns
        -------
        dict
            Dictionary containing MAE, MSE, RMSE, and R²
        """
        predictions = self.tree.predict(self.X)

        mae = mean_absolute_error(self.y, predictions)
        mse = mean_squared_error(self.y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y, predictions)

        metrics = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }

        print("\nRule-based Model Performance (Extracted Rules):")
        print(f"MAE : {mae:.4f}")
        print(f"MSE : {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²  : {r2:.4f}")

        return metrics


def main():
    """
    Main execution pipeline for Part-2 rule-based model extraction.
    """

    DATASET_PATH = "problem_58/dataset_58.csv"
    TARGET_PATH = "problem_58/target_58.csv"

    extractor = RuleBasedModelExtractor(
        dataset_path=DATASET_PATH,
        target_path=TARGET_PATH
    )

    extractor.load_data()
    extractor.analyze_feature_correlation()

    extractor.select_features([
        "feat_203",
        "feat_74",
        "feat_131",
        "feat_132"
    ])

    extractor.train_shallow_tree(max_depth=3, min_samples_leaf=50)
    extractor.print_extracted_rules()
    extractor.visualize_tree()
    extractor.evaluate_model()


if __name__ == "__main__":
    main()