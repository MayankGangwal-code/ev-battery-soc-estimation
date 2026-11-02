"""Machine-learning SOC estimators (scikit-learn pipelines on the causal features)."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def build_models(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    return {
        "Gradient Boosting": Pipeline(
            [
                (
                    "reg",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05, max_iter=600, max_leaf_nodes=31, l2_regularization=1.0, random_state=random_state
                    ),
                )
            ]
        ),
        "Neural Network": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "reg",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 64), activation="relu", learning_rate_init=1e-3, batch_size=512,
                        max_iter=150, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
