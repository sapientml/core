# sapientml-core

[![PyPI version](https://badge.fury.io/py/sapientml-core.svg)](https://badge.fury.io/py/sapientml-core)
[![Python Versions](https://img.shields.io/pypi/pyversions/sapientml-core.svg)](https://pypi.org/project/sapientml-core)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

`sapientml-core` is the default plugin for [SapientML](https://github.com/sapientml/sapientml).  
It automatically selects preprocessing steps and machine learning models based on dataset meta-features, and generates executable Python scripts.

## Overview

SapientML is a code-generation AutoML framework. `sapientml-core` provides the core pipeline generation logic.

```
Input Dataset
      │
      ▼
Meta-feature Extraction (DatasetSummary)
      │
      ▼
Preprocessing Label Prediction (pp_models.pkl)
ML Model Prediction            (mp_model_1/2.pkl)
      │
      ▼
Template-based Code Generation (Adaptation)
      │
      ▼
Candidate Script Execution & Evaluation
      │
      ▼
Best Script Output (final_script.py / final_train.py / final_predict.py)
```

### Plugin Architecture

The package registers itself with the SapientML framework via entry points defined in `pyproject.toml`.

| Group | Key | Class |
|---|---|---|
| `sapientml.pipeline_generator` | `sapientml` | `SapientMLGenerator` |
| `sapientml.config` | `sapientml` | `SapientMLConfig` |
| `sapientml.datastore` | `localfile` | `LocalFile` |
| `sapientml.preprocess` | `default` | `DefaultPreprocess` |
| `sapientml.export_modules` | `sample-dataset` | `datastore.localfile.export_modules` |

## Installation

```bash
pip install sapientml-core
```

> **System Requirements**  
> MeCab is required for Japanese text processing.
> ```bash
> # Ubuntu / Debian
> sudo apt-get install -y mecab libmecab-dev mecab-ipadic-utf8
> # macOS
> brew install mecab mecab-ipadic
> ```

### Install from Source

```bash
git clone https://github.com/sapientml/core.git
cd core
pip install uv
uv sync
```

## Quick Start

```python
import pandas as pd
from sapientml import SapientML

df_train = pd.read_csv("train.csv")
df_test  = pd.read_csv("test.csv")

# sapientml-core is used by default
sml = SapientML(
    target_columns=["target"],
    task_type="classification",   # "classification" or "regression"
    adaptation_metric="f1",       # metric to optimize
)

# Generate, execute, and select the best pipeline
sml.fit(df_train, output_dir="./outputs")

# Predict
predictions = sml.predict(df_test)
```

Generated output files:

| File | Description |
|---|---|
| `final_script.py` | Test script for the best model |
| `final_train.py` | Training script for the best model |
| `final_predict.py` | Inference script for the best model |
| `{N}_script.py` | N-th candidate script |
| `final_script.out.json` | Score and hyperparameter details |

## SapientMLConfig Parameters

The following `sapientml-core`-specific options can be passed as constructor arguments to `SapientML()`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_models` | `int` | `3` | Number of candidate models to generate and evaluate (max 30) |
| `seed_for_model` | `int` | `42` | Random seed for models |
| `id_columns_for_prediction` | `list[str]` | `None` | Column names to include in prediction output as identifiers |
| `use_word_list` | `list[str]` \| `dict[str, list[str]]` | `None` | Word list used when generating features from text columns |
| `hyperparameter_tuning` | `bool` | `False` | Enable hyperparameter tuning via Optuna |
| `hyperparameter_tuning_n_trials` | `int` | `10` | Number of Optuna trials |
| `hyperparameter_tuning_timeout` | `int` | `0` | Time limit for HPO per script in seconds (0 = unlimited) |
| `hyperparameter_tuning_random_state` | `int` | `1023` | Random seed for hyperparameter tuning |
| `predict_option` | `"default"` \| `"probability"` \| `None` | `None` | Prediction method override (`None` = follow metric requirements) |
| `permutation_importance` | `bool` | `True` | Include permutation importance calculation code in output |
| `add_explanation` | `bool` | `False` | Generate EDA and explanation notebooks (`.ipynb`) |
| `export_preprocess_dataset` | `bool` | `False` | Export the preprocessed dataset |

### Example: With Hyperparameter Tuning

```python
sml = SapientML(
    target_columns=["price"],
    task_type="regression",
    adaptation_metric="r2",
    n_models=5,
    hyperparameter_tuning=True,
    hyperparameter_tuning_n_trials=50,
    hyperparameter_tuning_timeout=300,
    add_explanation=True,
)
sml.fit(df_train, output_dir="./outputs")
```

## Supported Models and Preprocessing

### Machine Learning Models

| Task | Models |
|---|---|
| Classification & Regression | RandomForest, ExtraTrees, LightGBM, XGBoost, CatBoost, GradientBoosting, AdaBoost, DecisionTree, SVM, LinearSVM, LogisticRegression / LinearRegression, SGD, MLP, Lasso |
| Classification only | MultinomialNB, GaussianNB, BernoulliNB |

### Preprocessing Components

| Category | Processing |
|---|---|
| Missing value imputation | Per-column imputation for numeric and string columns |
| Categorical encoding | One-Hot encoding, Label encoding |
| Scaling | StandardScaler |
| Text processing | CountVectorizer, TF-IDF, MeCab (Japanese), langdetect (language detection) |
| Date handling | Numeric conversion of date columns |
| Class imbalance | SMOTE |
| Log transformation | log1p applied to target columns |

## Retraining Meta-learning Models

To retrain the bundled prediction models (`.pkl`) on a custom corpus:

```bash
# Install additional training dependencies
pip install -r requirements-training.txt

# Run the 5-step meta-learning pipeline
python -c "
from sapientml_core import SapientMLGenerator
gen = SapientMLGenerator()
gen.train(tag='my_experiment', num_parallelization=200)
"
```

Training consists of five steps:

1. **Static analysis & dataset snapshot extraction**
2. **Data augmentation via mutation**
3. **Meta-feature extraction**
4. **Preprocessing predictor & ML model training** (scikit-learn DecisionTree / Logistic / SVC)
5. **Dataflow model construction** (label dependency and ordering)

Results are saved to `sapientml_core/.cache/[tag]/`.

## Development

### Setup

```bash
uv sync --group dev
uv run pre-commit install
```

### Linting

```bash
uv run pysen run lint
# Auto-fix
uv run pysen run format
```

### Testing

```bash
uv run pytest
```

Coverage is reported automatically via `--cov=sapientml_core` (configured in `pytest.ini_options`).

## Python Version Support

| Version | Supported | Models Used |
|---|---|---|
| 3.9 | ✅ | `models/PY39/` |
| 3.10 | ✅ | `models/PY310/` |
| 3.11 | ✅ | `models/PY311/` |
| 3.12 | ✅ | `models/PY311/` (clamped to newest) |
| 3.13 | ✅ | `models/PY311/` (clamped to newest) |

## License

[Apache License 2.0](LICENSE)

## Related Repositories

- [sapientml/sapientml](https://github.com/sapientml/sapientml) — SapientML core framework
- [sapientml/sapientml-core](https://pypi.org/project/sapientml-core/) — PyPI package
