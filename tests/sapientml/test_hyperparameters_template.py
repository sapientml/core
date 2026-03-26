# Copyright 2023-2024 The SapientML Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for sapientml/core#63.

Verify that:
1. hyperparameters.py.jinja renders ``suggest_float('alpha', 1e-6, 1e-3, log=True)``
   for MLPClassifier and MLPRegressor — the old upper bound of 1.0 caused
   "ValueError: Solver produced non-finite parameter weights".
2. No model in the template uses the deprecated ``suggest_loguniform()`` API
   (removed in optuna v4+).
3. An optuna study using the fixed alpha range on synthetic data completes all
   trials without ValueError for both MLPClassifier and MLPRegressor.
"""

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
from jinja2 import Environment, FileSystemLoader
from sklearn.neural_network import MLPClassifier, MLPRegressor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).parent.parent.parent / "sapientml_core" / "templates"

# Every model name that appears in hyperparameters.py.jinja
_ALL_MODEL_NAMES = [
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "BernoulliNB",
    "CatBoostClassifier",
    "CatBoostRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "GaussianNB",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "Lasso",
    "LGBMClassifier",
    "LGBMRegressor",
    "LinearRegression",
    "LinearSVC",
    "LinearSVR",
    "LogisticRegression",
    "MLPClassifier",
    "MLPRegressor",
    "MultinomialNB",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "SGDClassifier",
    "SGDRegressor",
    "SVC",
    "SVR",
    "XGBClassifier",
    "XGBRegressor",
]


def _render(model_name: str) -> str:
    env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), trim_blocks=True)
    return env.get_template("model_templates/hyperparameters.py.jinja").render(model_name=model_name)


# ---------------------------------------------------------------------------
# Template content — MLP alpha search range (issue #63)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["MLPClassifier", "MLPRegressor"])
def test_misc_hyperparameters_mlp_alpha_uses_narrowed_upper_bound(model_name):
    """Fix #63: alpha must be sampled from [1e-6, 1e-3], not [1e-6, 1.0].

    An upper bound of 1.0 caused optuna to sample large alpha values that
    drove the solver to produce non-finite weights.
    """
    code = _render(model_name)
    assert "suggest_float('alpha', 1e-6, 1e-3, log=True)" in code


# ---------------------------------------------------------------------------
# Template content — deprecated suggest_loguniform (issue #63)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", _ALL_MODEL_NAMES)
def test_misc_hyperparameters_no_suggest_loguniform_any_model(model_name):
    """suggest_loguniform() was removed in optuna v4; no model may use it."""
    assert "suggest_loguniform" not in _render(model_name)


# ---------------------------------------------------------------------------
# Integration — optuna study with fixed alpha range must not raise ValueError
# ---------------------------------------------------------------------------


def _synthetic_dataset(task_type: str):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((120, 8)))
    y = pd.Series(rng.integers(0, 2, 120).astype(int) if task_type == "classification" else rng.standard_normal(120))
    return X.iloc[:90], X.iloc[90:], y.iloc[:90], y.iloc[90:]


@pytest.mark.parametrize(
    "model_cls,task_type",
    [
        (MLPRegressor, "regression"),
        (MLPClassifier, "classification"),
    ],
)
def test_misc_mlp_tuning_fixed_alpha_range_completes_without_error(model_cls, task_type):
    """Fix #63: all optuna trials with alpha in [1e-6, 1e-3] must complete.

    Mirrors what the rendered hyperparameter-tuning script does at runtime:
    an objective that calls ``suggest_float('alpha', 1e-6, 1e-3, log=True)``
    and fits the MLP must never raise ValueError.
    """
    X_train, X_test, y_train, y_test = _synthetic_dataset(task_type)

    def objective(trial):
        params = {
            "activation": trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"]),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "sgd", "adam"]),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-3, log=True),  # fixed range from PR #119
        }
        model = model_cls(random_state=0, max_iter=500, **params)
        model.fit(X_train, y_train)
        return model.score(X_test, y_test)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    study.optimize(objective, n_trials=10, timeout=60)

    assert len(study.trials) > 0, "No trials ran"
    failed = [t for t in study.trials if t.state != optuna.trial.TrialState.COMPLETE]
    assert not failed, f"{len(failed)} trial(s) did not complete: {[t.state for t in failed]}"
