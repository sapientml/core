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

"""Regression tests for sapientml/core#63 and sapientml/core#64.

Verify that:
1. hyperparameters.py.jinja renders ``suggest_float('alpha', 1e-6, 1e-3, log=True)``
   for MLPClassifier and MLPRegressor — the old upper bound of 1.0 caused
   "ValueError: Solver produced non-finite parameter weights".
2. No model in the template uses the deprecated ``suggest_loguniform()`` API
   (removed in optuna v4+).
3. An optuna study using the fixed alpha range on synthetic data completes all
   trials without ValueError for both MLPClassifier and MLPRegressor.
4. GradientBoostingClassifier never includes 'exponential' loss when is_multiclass=True
   — sklearn raises "ExponentialLoss requires 2 classes" for multiclass targets.
5. 'deviance' (removed in sklearn 1.3) never appears in the rendered template.
"""

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
from jinja2 import Environment, FileSystemLoader
from sklearn.ensemble import GradientBoostingClassifier
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


def _render(model_name: str, is_multiclass: bool = False) -> str:
    env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), trim_blocks=True)
    return env.get_template("model_templates/hyperparameters.py.jinja").render(
        model_name=model_name, is_multiclass=is_multiclass
    )


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


# ---------------------------------------------------------------------------
# Template content — GradientBoostingClassifier multiclass loss (issue #64)
# ---------------------------------------------------------------------------


def test_gradient_boosting_classifier_multiclass_excludes_exponential():
    """Fix #64: 'exponential' must not appear when is_multiclass=True.

    sklearn raises "ExponentialLoss requires 2 classes" for any target with
    more than two classes, so the template must guard that candidate.
    """
    code = _render("GradientBoostingClassifier", is_multiclass=True)
    assert "exponential" not in code


def test_gradient_boosting_classifier_binary_includes_exponential():
    """Fix #64: 'exponential' must be available for binary classification."""
    code = _render("GradientBoostingClassifier", is_multiclass=False)
    assert "exponential" in code


def test_gradient_boosting_classifier_always_excludes_deviance():
    """'deviance' was removed in sklearn 1.3 and must never appear in the template."""
    for is_multiclass in (False, True):
        code = _render("GradientBoostingClassifier", is_multiclass=is_multiclass)
        assert "deviance" not in code, f"'deviance' found with is_multiclass={is_multiclass}"


# ---------------------------------------------------------------------------
# Integration — optuna study with multiclass GradientBoostingClassifier (issue #64)
# ---------------------------------------------------------------------------


def _synthetic_multiclass_dataset():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((150, 4)))
    y = pd.Series(rng.integers(0, 3, 150).astype(int))  # 3 classes (like Iris)
    return X.iloc[:112], X.iloc[112:], y.iloc[:112], y.iloc[112:]


def test_gradient_boosting_multiclass_tuning_completes_without_error():
    """Fix #64: optuna trials for GradientBoostingClassifier on 3-class data must not
    raise 'ExponentialLoss requires 2 classes'.

    Uses only loss candidates that are valid for multiclass (log_loss).
    """
    X_train, X_test, y_train, y_test = _synthetic_multiclass_dataset()

    def objective(trial):
        params = {
            "loss": trial.suggest_categorical("loss", ["log_loss"]),
            "n_estimators": trial.suggest_int("n_estimators", 10, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        model = GradientBoostingClassifier(random_state=0, **params)
        model.fit(X_train, y_train)
        return model.score(X_test, y_test)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1))
    study.optimize(objective, n_trials=5, timeout=30)

    assert len(study.trials) > 0, "No trials ran"
    failed = [t for t in study.trials if t.state != optuna.trial.TrialState.COMPLETE]
    assert not failed, f"{len(failed)} trial(s) failed: {[t.state for t in failed]}"
