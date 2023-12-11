# Copyright 2023 The SapientML Authors
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

import re
from collections import defaultdict
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
from pydantic import BaseModel, Field, field_validator
from sapientml.params import Code, Config, Task

from . import ps_macros
from .meta_features import (
    MetaFeatures,
    generate_column_meta_features,
    generate_model_meta_features,
    generate_pp_meta_features,
)

PipelineSkeleton = (
    dict  # dict[str, Union[float, dict[str, Union[float, list[str], list[dict[str, Union[float, int, str]]]]]]]
)

MAX_NUM_OF_COLUMNS = 10000000
MAX_SEED = 2**32 - 1
MAX_COLUMN_NAME_LENGTH = 1000
MAX_HPO_N_TRIALS = 100000
MAX_HPO_TIMEOUT = 500000
MAX_N_MODELS = 30
INITIAL_TIMEOUT = 600


class SapientMLConfig(Config):
    """
    Configuration arguments for SapientMLGenerator.

    Attributes
    ----------
    n_models: int, default 3
        Number of output models to be tried.
    seed_for_model: int, default 42
        Random seed for models such as RandomForestClassifier.
    id_columns_for_prediction: Optional[list[str]], default None
        Name of the dataframe columns that outputs the prediction result.
    use_word_list: Optional[Union[list[str], dict[str, list[str]]]], default None
        List of words to be used as features when generating explanatory variables from text.
        If dict type is specified, key must be a column name and value must be a list of words.
    hyperparameter_tuning: bool, default False
        On/Off of hyperparameter tuning.
    hyperparameter_tuning_n_trials: int, default 10
        The number of trials of hyperparameter tuning.
    hyperparameter_tuning_timeout: int, default 0
        Time limit for hyperparameter tuning in each generated script.
        Ignored when hyperparameter_tuning is False.
    hyperparameter_tuning_random_state: int, default 1023
        Random seed for hyperparameter tuning.
    predict_option: Literal["default", "probability", None], default None
        Specify predict method (default: predict(), probability: predict_proba(), None: Comply with metric requirements.)
    permutation_importance: bool, default True
        On/Off of outputting permutation importance calculation code.
    add_explanation: bool, default False
        If True, outputs ipynb files including EDA and explanation.

    """

    n_models: int = 3
    seed_for_model: int = 42
    id_columns_for_prediction: Optional[list[str]] = None
    use_word_list: Optional[Union[list[str], dict[str, list[str]]]] = None
    hyperparameter_tuning: bool = False
    hyperparameter_tuning_n_trials: int = 10
    hyperparameter_tuning_timeout: int = 0
    hyperparameter_tuning_random_state: int = 1023
    predict_option: Optional[Literal["default", "probability"]] = None
    permutation_importance: bool = True
    add_explanation: bool = False

    def postinit(self):
        """Set initial_timeout and hyperparameter_tuning_timeout.

        If initial_timeout is set as None and hyperparameter_tuning is false, set initial_timeout as INITIAL_TIMEOUT.

        For hyperparameter_tuning_timeout,
        if both initial_timeout and hyperparameter_tuning_timeout are set as None, set hyperparameter_tuning_timeout as INITIAL_TIMEOUT.

        If initial_timeout is set and hyperparameter_tuning is True,
        and hyperparameter_tuning_timeout is None :
            Set the hyperparameter_tuning_timeout to unlimited.(hyperparameter_tuning_timeout = self.initial_timeout.)
            Since initial_timeout always precedes hyperparameter_tuning_timeout,
            it can be expressed that there is no time limit for hyperparameters during actual execution.
        """
        if self.id_columns_for_prediction is None:
            self.id_columns_for_prediction = []

        if self.initial_timeout is None:
            if self.hyperparameter_tuning:
                self.initial_timeout = 0
            else:
                self.initial_timeout = INITIAL_TIMEOUT
            if self.hyperparameter_tuning_timeout is None:
                self.hyperparameter_tuning_timeout = INITIAL_TIMEOUT
        elif self.hyperparameter_tuning_timeout is None:
            if self.hyperparameter_tuning:
                self.hyperparameter_tuning_timeout = self.initial_timeout
            else:
                self.hyperparameter_tuning_timeout = INITIAL_TIMEOUT

    @field_validator("n_models")
    def _check_n_models(cls, v):
        if v <= 0 or MAX_N_MODELS < v:
            raise ValueError(f"{v} is out of [1, {MAX_N_MODELS}]")
        return v

    @field_validator(
        "id_columns_for_prediction",
        "use_word_list",
    )
    def _check_num_of_column_names(cls, v):
        if v is None:
            return v
        if len(v.keys() if isinstance(v, dict) else v) >= MAX_NUM_OF_COLUMNS:
            raise ValueError(f"The number of columns must be smaller than {MAX_NUM_OF_COLUMNS}")
        return v

    @field_validator(
        "id_columns_for_prediction",
        "use_word_list",
    )
    def _check_column_name_length(cls, v):
        if v is None:
            return v
        for _v in v.keys() if isinstance(v, dict) else v:
            if len(_v) >= MAX_COLUMN_NAME_LENGTH:
                raise ValueError(f"Column name length must be shorter than {MAX_COLUMN_NAME_LENGTH}")
        return v

    @field_validator("seed_for_model")
    def _check_seed(cls, v):
        if v < 0 or MAX_SEED < v:
            raise ValueError(f"{v} is out of [0, {MAX_SEED}]")
        return v

    @field_validator("hyperparameter_tuning_n_trials")
    def _check_hyperparameter_tuning_n_trials(cls, v):
        if v < 1 or MAX_HPO_N_TRIALS < v:
            raise ValueError(f"{v} is out of [1, {MAX_HPO_N_TRIALS}]")
        return v

    @field_validator("hyperparameter_tuning_timeout")
    def _check_hyperparameter_tuning_timeout(cls, v):
        if v < 0 or MAX_HPO_TIMEOUT < v:
            raise ValueError(f"{v} is out of [0, {MAX_HPO_TIMEOUT}]")
        return v

    @field_validator("hyperparameter_tuning_random_state")
    def _check_hyperparameter_tuning_random_state(cls, v):
        if v < 0 or MAX_SEED < v:
            raise ValueError(f"{v} is out of [0, {MAX_SEED}]")
        return v


class Column(BaseModel):
    """
    Describing meta-features of a column in the input dataset.

    Attributes
    ----------
    dtype : str
        Data type of the column of features.
    meta_features : MetaFeatures | None
        Meta features of the column.
    has_negative_value : bool
        Whether the column has a negative value.
    """

    dtype: str
    meta_features: Optional[MetaFeatures]
    has_negative_value: bool

    @field_validator("dtype")
    def check_dtype(cls, v):
        if len(v) > 100:
            raise ValueError(f"'{v}' is invalid as a dtype")
        return v

    @field_validator("meta_features")
    def check_meta_features(cls, v):
        if v is None:
            return v
        for k, _v in v.items():
            if not re.match(r"feature:[a-z_0-9]+", k):
                raise ValueError(f"'{k}' is invalid as a feature name")
            if isinstance(_v, int):
                pass
            elif isinstance(_v, float):
                pass
            elif isinstance(_v, str) and len(_v) >= MAX_COLUMN_NAME_LENGTH:
                raise ValueError(f"Meta feature string value must be shorter than {MAX_COLUMN_NAME_LENGTH}")
            elif isinstance(_v, list):
                for s in _v:
                    if not isinstance(s, str):
                        raise ValueError("The list must contain string values only")
                    elif len(s) >= MAX_COLUMN_NAME_LENGTH:
                        raise ValueError(
                            f"Column name length in meta features must be shorter than {MAX_COLUMN_NAME_LENGTH}"
                        )
        return v


class DatasetSummary(BaseModel):
    """Describing meta-features of datasets.

    Attributes
    ----------
    columns : dict[str, Column]
        Dictionary of meta features.
        - dict[column_name:str : [dtype:str , meta_features:MetaFeatures | None , has_negative_value:bool]]
    meta_features_pp : MetaFeatures
        Meta-features used to consider preprocessing components.
    meta_features_m : MetaFeatures
        Meta-features used to consider machine learning models.
    has_multi_class_targets : bool
        Whether has multi class targets.
    has_inf_value_targets : bool
        Whether has inf value targets.
    cols_almost_missing_string : list[str] | None, default None
        List of string columns where almost all values are missing.
    cols_almost_missing_numeric : list[str] | None, default None
        List of numerical columns where almost all values are missing.
    cols_str_other : list[str] | None, default None
        List of string columns which are neither categorical nor text columns.
    """

    columns: dict[str, Column]
    meta_features_pp: MetaFeatures
    meta_features_m: MetaFeatures
    has_multi_class_targets: bool
    has_inf_value_targets: bool
    cols_almost_missing_string: Optional[list[str]] = None
    cols_almost_missing_numeric: Optional[list[str]] = None
    cols_str_other: Optional[list[str]] = None

    @field_validator("columns", "cols_almost_missing_string", "cols_almost_missing_numeric", "cols_str_other")
    def check_num_of_columns(cls, v):
        if v is None:
            return v
        if len(v.keys() if isinstance(v, dict) else v) >= MAX_NUM_OF_COLUMNS:
            raise ValueError(f"The number of columns must be smaller than {MAX_NUM_OF_COLUMNS}")
        return v

    @field_validator("columns", "cols_almost_missing_string", "cols_almost_missing_numeric", "cols_str_other")
    def check_column_name_length(cls, v):
        if v is None:
            return v
        for _v in v.keys() if isinstance(v, dict) else v:
            if len(_v) >= MAX_COLUMN_NAME_LENGTH:
                raise ValueError(f"Column name length must be shorter than {MAX_COLUMN_NAME_LENGTH}")
        return v

    @field_validator("meta_features_pp", "meta_features_m")
    def check_meta_features(cls, v):
        for k, _v in v.items():
            if not re.match(r"feature:[a-z_0-9]+", k):
                raise ValueError(f"'{k}' is invalid as a feature name")
            if isinstance(_v, int):
                pass
            elif isinstance(_v, float):
                pass
            elif isinstance(_v, str) and len(_v) >= MAX_COLUMN_NAME_LENGTH:
                raise ValueError(f"Meta feature string value must be shorter than {MAX_COLUMN_NAME_LENGTH}")
            elif isinstance(_v, list):
                for s in _v:
                    if not isinstance(s, str):
                        raise ValueError("The list must contain string values only")
                    elif len(s) >= MAX_COLUMN_NAME_LENGTH:
                        raise ValueError(
                            f"Column name length in meta features must be shorter than {MAX_COLUMN_NAME_LENGTH}"
                        )
        return v


class ModelLabel(BaseModel):
    """Describing a model name and information to embed it to the code.

    Attributes
    ----------
    label_name: str
        Label name.
    predict_proba : bool, default False
        Whether to predict probability.
    hyperparameters : Any | None, default None
        Hyperparameters.
    meta_features : list[Any], default Field(default_factory=list)
        List of Meta features.
    """

    label_name: str
    predict_proba: bool = False
    hyperparameters: Optional[Any] = None
    meta_features: list[Any] = Field(default_factory=list)

    def __str__(self):
        return self.label_name

    def __repr__(self):
        return str(self)


class SimplePipeline(Code):
    """Generated code with information during adaptation.

    Attributes
    ----------
    pipeline_json : dict, dafault Field(default_factory=lambda: defaultdict(dict))
        Dictionary of pipeline.
    labels : PipelineSkeleton | None , dafault None
        PipelineSkeleton class.
    model : ModelLabel | None , dafault None
        ModelLabel class.
    """

    # pipeline json
    pipeline_json: dict = Field(default_factory=lambda: defaultdict(dict))

    labels: Optional[PipelineSkeleton] = None
    model: Optional[ModelLabel] = None


class Pipeline(SimplePipeline):
    """Generated code with information used internally during adaptation.

    Attributes
    ----------
    task : Task
        Object of the Task class.
    dataset_summary : DatasetSummary
        Object of the DatasetSummary Class.
    config : SapientMLConfig
        Object of the SapientMLConfig.
    adaptation_metric : str | None, default None
        Adaptation metric.
    all_columns_datatypes : dict, default Field(default_factory=dict)
        Dictionary of all column data types.
    inverse_target : bool, default False
        Whether expm1 is required for target columns because log1p is already applied to them.
    sparse_matrix : bool, default False
        Whether the data is converted to sparse matrix in the pipeline.
    train_column_names : list[str], default Field(default_factory=list)
        Column name of training.
    test_column_names : list[str], default Field(default_factory=list)
        Column name of testing.
    is_multi_class_multi_targets : bool, default False
        Whether is multi-class and multi-target.
        - To handle following case;
            - metrics : Accuracy
            - task    : multi-class and multi-targets
        - because sklearn.accuracy_score doesn't support multi-class classification
          involving multiple columns.
    id_columns_for_prediction : list[str], default Field(default_factory=list)
        ID columns for prediction.
    output_dir_path : str, default ""
        Output directory path.
    """

    task: Task
    dataset_summary: DatasetSummary
    config: SapientMLConfig
    adaptation_metric: Optional[str] = None
    all_columns_datatypes: dict = Field(default_factory=dict)
    inverse_target: bool = False
    sparse_matrix: bool = False  # Whether the data is converted to sparse matrix in the pipeline
    train_column_names: list[str] = Field(default_factory=list)
    test_column_names: list[str] = Field(default_factory=list)

    # To handle following case;
    #   metrics : Accuracy
    #   task    : multi-class and multi-targets
    # because sklearn.accuracy_score doesn't support multi-class
    is_multi_class_multi_targets: bool = False

    id_columns_for_prediction: list[str] = Field(default_factory=list)
    output_dir_path: str = ""


def summarize_dataset(df_train: pd.DataFrame, task: Task) -> DatasetSummary:
    """Summarize dataset.

    Parameters
    ----------
        df_train : pd.DataFrame
            Input dataset.
        task : Task
            Object of the Task class.

    Returns
    ----------
        DatasetSummary
    """
    is_multi_classes: list[bool] = []
    for target in task.target_columns:
        is_multi_classes.append(len(df_train[target].unique()) > 1)
    has_multi_class_targets = all(is_multi_classes)

    has_inf_value_targets = bool(df_train[task.target_columns].isin([np.inf, -np.inf]).sum().sum() > 0)

    # handle almost_missing columns
    cols_all_missing = df_train.columns[df_train.isnull().all()]
    cols_almost_missing = df_train.columns[df_train.isna().sum() / len(df_train) > 0.8]
    cols_almost_missing = sorted(list(set(cols_almost_missing) - set(cols_all_missing)))
    cols_almost_missing_numeric = [col for col in cols_almost_missing if is_numeric_dtype(df_train[col])]
    cols_almost_missing_string = [col for col in cols_almost_missing if not is_numeric_dtype(df_train[col])]

    columns = dict()
    for column_name in df_train.columns:
        meta_features = generate_column_meta_features(df_train[[column_name]])

        columns[column_name] = Column(
            dtype=str(df_train[column_name].dtype),
            meta_features=meta_features,
            has_negative_value=bool(
                is_numeric_dtype(df_train[column_name]) and (df_train[[column_name]].values <= 0).any()
            ),
        )

    # Generate the meta-features
    meta_features_pp = generate_pp_meta_features(df_train, task.target_columns)
    if ps_macros.STR_OTHER in meta_features_pp:
        cols_str_other = meta_features_pp[ps_macros.STR_OTHER]
        del meta_features_pp[ps_macros.STR_OTHER]

    is_clf_task = 1 if task.task_type == "classification" else 0
    meta_features_m = generate_model_meta_features(df_train, task.target_columns, is_clf_task)

    return DatasetSummary(
        columns=columns,
        meta_features_pp=meta_features_pp,
        meta_features_m=meta_features_m,
        has_multi_class_targets=has_multi_class_targets,
        has_inf_value_targets=has_inf_value_targets,
        cols_almost_missing_string=cols_almost_missing_string,
        cols_almost_missing_numeric=cols_almost_missing_numeric,
        cols_str_other=cols_str_other,
    )
