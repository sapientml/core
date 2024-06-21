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

import ast
import json
import os
import time
from glob import glob
from pathlib import Path
from threading import Thread
from typing import Optional

import nbformat
from nbconvert.preprocessors.execute import ExecutePreprocessor
from nbformat import NotebookNode
from sapientml.params import CancellationToken
from sapientml.util.logging import setup_logger

from .code_template import Code_Template
from .pipeline_explanation import Pipeline_Explanation

# To surpress warinings when executing ipynb
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"


class AST_Update:
    """AST_Update class."""

    def __init__(self, visualization_code=None, logger=None, local_path=None):
        self.visualization_code = visualization_code
        self.logger = logger
        self.local_path = local_path
        if logger is None:
            self.logger = setup_logger()

        self.FUNCION_MAPPING = {
            "STRING_COLS_WITH_MISSING_VALUES": "__cat_missing_data__",
            "NUMERIC_COLS_WITH_MISSING_VALUES": "__num_missing_data__",
            "NUMERIC_COLS_TO_SCALE =": "__num_to_scale__",
            "# OUTPUT PREDICTION": "__set_prediction__",
            "TARGET_COLUMNS =": "__set_target__",
            "model =": "__model__",
            "ordinal_encoder = OrdinalEncoder": "__OrdinalEncoder_exp__",
            "irrelevant_columns =": "__drop_Irrelevant__",
            "def process_text": "__process_text_exp__",
            "tfidfvectorizer =": "__tfidfvectorizer_exp__",
            "y_pred = model.predict_proba": "__predict_proba_exp__",
            "smote = SMOTE()": "__smote__exp__",
            "standard_scaler = StandardScaler()": "__standard_scaler__exp__",
            "simple_imputer = SimpleImputer(": "__simple_imputer__exp__",
            # The followings all mean "after #LOAD DATA block"
            "# TRAIN-TEST SPLIT": "__test_dataset_prediction_columns__exp__",
            "# HANDLE MIXED TYPE": "__test_dataset_prediction_columns__exp__",
            "# CONVERT INF TO NAN": "__test_dataset_prediction_columns__exp__",
            "# HANDLE JAPANESE TEXT": "__test_dataset_prediction_columns__exp__",
            "# HANDLE ITERABLE VALUES IN DATAFRAME": "__test_dataset_prediction_columns__exp__",
            "# STORE PREDICTION RELEVANT COLUMNS": "__test_dataset_prediction_columns__exp__",
            "# PREPROCESSING-1": "__test_dataset_prediction_columns__exp__",
            "# DETACH TARGET": "__test_dataset_prediction_columns__exp__",
            "# Remove special symbols": "__test_dataset_prediction_columns__exp__",
            "# DROP IGNORED COLUMNS": "__test_dataset_prediction_columns__exp__",
            "# SET ID_COLUMNS TO DATAFRAME'S INDEX": "__test_dataset_prediction_columns__exp__",
            "# Confusion Matrix": "__set_confusion_matrix__",
            "# Shap": "__set_shap__",
        }

        # add functionname:Bool to show the function can be added as duplicate
        self.FUNCION_MAPPING_DUPLICATE_FLAG = {
            "__model__": False,
            "__standard_scaler__exp__": False,
            "__simple_imputer__exp__": False,
            "__smote__exp__": False,
            "__test_dataset_prediction_columns__exp__": False,
        }
        self.last_status = {}

    def process(self, loc, prev):
        """process method.

        Parameters
        ----------
        loc : str
            A line in block code from jupyter content template.
        prev : str
            A line in block code from jupyter content template.

        """
        for func in self.FUNCION_MAPPING:
            # run loc against all function calls
            # if there is a match, the it will be executed
            if loc.strip().startswith(func):
                try:
                    execute = False
                    fired_func = self.FUNCION_MAPPING[func]
                    if not self.FUNCION_MAPPING_DUPLICATE_FLAG.get(fired_func, True):
                        # there is flag for fired_function
                        if not self.last_status.get(fired_func, False):
                            # previously the flag didn't changed
                            self.last_status[fired_func] = True
                            execute = True
                    else:
                        # there isn't any flag for fired_function
                        execute = True

                    if execute:
                        return self.__getattribute__(fired_func)(loc.strip(), prev)
                except AttributeError:
                    raise NotImplementedError(f"{fired_func}")

    def __model__(self, loc, prev):
        added_codes = []
        added_codes.append(
            (
                [
                    "# Training and Prediction",
                    "First, we will train a model based on the pre-processed training dataset.",
                    "Second, let's predict test values based on the trained model.",
                ],
                "markdown",
            )
        )
        if "model = CatBoostRegressor" in loc:
            added_codes.append(
                (
                    [
                        "## CatBoost Regression",
                        "We will use *CatBoostRegressor* which is a fast, scalable, high performance gradient boosting on decision trees library. Used for ranking, classification, regression and other ML tasks.",
                        "More details about *CatBoostRegressor* can be found [here](https://catboost.ai/docs/installation/python-installation-method-pip-install).",
                    ],
                    "markdown",
                )
            )
        elif "model = RandomForestClassifier" in loc:
            added_codes.append(
                (
                    [
                        "## Random Forest Classifier",
                        "We will use *RandomForestClassifier* which is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.",
                        "The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.",
                        "More details about *RandomForestClassifier* can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).",
                    ],
                    "markdown",
                )
            )
        elif "model = CatBoostClassifier" in loc:
            added_codes.append(
                (
                    [
                        "## Cat Boost Classifier",
                        "We will use *CatBoostClassifier* which is Training and applying models for the classification problems. It provides compatibility with the scikit-learn tools.",
                        "More details about *CatBoostClassifier* can be found [here](https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier).",
                    ],
                    "markdown",
                )
            )
        elif "model = LGBMRegressor" in loc:
            added_codes.append(
                (
                    [
                        "## LightGBM Regressor",
                        "We will use *LightGBM Regressor* which is constructing a gradient boosting model. We will use *lightgbm* package.",
                        "More details about *LightGBM Regressor* can be found [here](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html).",
                    ],
                    "markdown",
                )
            )
        elif "model = XGBClassifier" in loc:
            added_codes.append(
                (
                    [
                        "## XGBClassifier",
                        "We will use *XGBClassifier* from XGBoost package which is used for supervised learning problems, where we use the training data (with multiple features) to predict a target variable.",
                        "More details about *XGBClassifier* can be found [here](https://xgboost.readthedocs.io/en/stable/tutorials/model.html).",
                    ],
                    "markdown",
                )
            )
        elif "model = LogisticRegression" in loc:
            added_codes.append(
                (
                    [
                        "## LogisticRegression",
                        """Logistic regression, despite its name, is a linear model for classification rather than regression. Logistic regression is also known in the literature as logit regression, maximum-entropy classification (MaxEnt) or the log-linear classifier.
                                 In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.""",
                        "More details about *LogisticRegression* can be found [here](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).",
                    ],
                    "markdown",
                )
            )
        elif "model = ExtraTreesRegressor" in loc:
            added_codes.append(
                (
                    [
                        "## ExtraTreesRegressor",
                        """The SK-Learn.ensemble module includes two averaging algorithms based on randomized decision trees:
                                     the RandomForest algorithm and the Extra-Trees method. Both algorithms are perturb-and-combine techniques specifically designed for trees.
                                     This means a diverse set of classifiers is created by introducing randomness in the classifier construction.
                                     The prediction of the ensemble is given as the averaged prediction of the individual classifiers.
                                     In extremely randomized trees (*ExtraTreesRegressor*), randomness goes one step further in the way splits are computed.
                                     As in random forests, a random subset of candidate features is used, but instead of looking for the most discriminative thresholds, thresholds are drawn at random for each candidate feature and the best of these randomly-generated thresholds is picked as the splitting rule. """,
                        "More details about *ExtraTreesRegressor* can be found [here](https://scikit-learn.org/stable/modules/ensemble.html#forest).",
                    ],
                    "markdown",
                )
            )
        elif "model = LinearRegression" in loc:
            added_codes.append(
                (
                    [
                        "## LinearRegression",
                        """LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.""",
                        "More details about *LinearRegression* can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).",
                    ],
                    "markdown",
                )
            )
        elif "model = DecisionTreeClassifier" in loc:
            added_codes.append(
                (
                    [
                        "## Decision Tree Classifier",
                        """Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression.
                                 The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
                                 A tree can be seen as a piecewise constant approximation.""",
                        "More details about *DecisionTreeClassifier* can be found [here](https://scikit-learn.org/stable/modules/tree.html#tree).",
                    ],
                    "markdown",
                )
            )
        elif "model = XGBRegressor" in loc:
            added_codes.append(
                (
                    [
                        "## XGBRegressor",
                        """ We will use *XGBRegressor* which is "Extreme Gradient Boosting" approach and it is an implementation of gradient boosting trees algorithm.
                                 The XGBoost is a popular supervised machine learning model with characteristics like computation speed, parallelization, and performance.""",
                        "More details about *XGBRegressor* can be found [here](https://xgboost.readthedocs.io/en/latest/index.html).",
                    ],
                    "markdown",
                )
            )

        return added_codes

    def __set_confusion_matrix__(self, loc, prev):
        added_codes = []
        added_codes.append(
            (
                [
                    "## Confusion Matrix",
                    "confusion_matrix shows how many predictions are correct and incorrect per class.",
                ],
                "markdown",
            )
        )
        return added_codes

    def __set_shap__(self, loc, prev):
        added_codes = []
        added_codes.append(
            (
                [
                    "## Shap visualization for model prediction",
                ],
                "markdown",
            )
        )
        return added_codes

    def __set_prediction__(self, loc, prev):
        added_codes = []
        added_codes.append(
            (
                [
                    "## Prediction File",
                    'Our prediction results will be output to "prediction_result.csv".',
                ],
                "markdown",
            )
        )
        return added_codes

    def __simple_imputer__exp__(self, loc, prev):
        added_codes = []
        added_codes.append(
            (
                [
                    "## Imputation Transformer",
                    "We will use *SimpleImputer* which is an imputation transformer for completing missing values.",
                    "We can use out-of-the-box imputation transformer from Scikit-Learn packages. The detail and the list of complete parameters can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html).",
                ],
                "markdown",
            )
        )
        return added_codes

    def __standard_scaler__exp__(self, loc, prev):
        added_codes = []
        added_codes.append(
            (
                [
                    "## Standard Scaler"
                    "\nWe will use Scikit-Learn *StandardScaler* which standardizes features by removing the mean and scaling to unit variance.",
                    "The deatil can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).",
                ],
                "markdown",
            )
        )
        return added_codes

    def __predict_proba_exp__(self, loc, prev):
        added_codes = []
        added_codes.append(
            (
                [
                    "We need to predict the probability for each example, therefore we will use **predict_proba()** that generates the probability.",
                ],
                "markdown",
            )
        )
        return added_codes

    def __smote__exp__(self, loc, prev):
        added_codes = []
        added_codes.append(
            (
                [
                    "## SMOTE"
                    "\nWe will use SMOTE (Synthetic Minority Over-sampling Technique) as presented by Chawla et al. (the paper is available [here](https://www.jair.org/index.php/jair/article/view/10302).)",
                    "Implementation and examples of SMOTE is available in [Imbalanced-Learn](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html#r001eabbe5dd7-1).",
                ],
                "markdown",
            )
        )
        return added_codes

    def __process_text_exp__(self, loc, prev):
        added_codes = []
        if "TEXT_COLUMNS" in prev:
            tree = ast.parse(prev)
            keys = [v.value for v in tree.body[0].value.elts]
            recs = ", ".join(keys)
            singular = "s" if len(keys) > 1 else ""
            if len(keys):
                added_codes.append(
                    (
                        [
                            "## Text Processing",
                            f"The dataset has **{len(keys)}** text value{singular} as follows: **{recs}**.",
                            "Now, let's covert the text as follows.\n",
                            "- First, convert text to lowercase;\n",
                            "- Second, strip all punctuations;\n",
                            "- Finally, convert all numbers in text to 'num'; therefore, in the next step our model will use a single token instead of valriety of tokens of numbers.",
                        ],
                        "markdown",
                    )
                )
        return added_codes

    def __tfidfvectorizer_exp__(self, loc, prev):
        added_codes = []
        added_codes.append(
            (
                [
                    "# Text Vectorizer",
                    "In the next step, we will transfer pre-processed text columns to a vector representation. The vector representation allows us to train a model based on numerical representations.",
                    "We will use TfidfVectorizer and more detail can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).",
                ],
                "markdown",
            )
        )
        return added_codes

    def __drop_Irrelevant__(self, loc, prev):
        added_codes = []
        try:
            tree = ast.parse(loc)
            keys = [v.value for v in tree.body[0].value.elts]
            recs = ", ".join(keys)
            singular = "s" if len(keys) > 1 else ""
            singular_verb = "are" if len(keys) > 1 else "is"
            added_codes.append(
                (
                    [
                        "### Discard Irrelevant Columns",
                        f"In the given input dataset there {singular_verb} **{len(keys)}** column{singular} that can be removed as follows: {recs}",
                    ],
                    "markdown",
                )
            )
        except Exception as ex:
            self.logger.warning(f"Column extraction issue ({ex})")
            return []
        return added_codes

    def __OrdinalEncoder_exp__(self, loc, prev):
        added_codes = []
        try:
            if "CATEGORICAL_COLS" in prev:
                tree = ast.parse(prev)
                keys = [v.value for v in tree.body[0].value.elts]
                recs = ", ".join(keys)
                singular = "s" if len(keys) > 1 else ""
                singular_verb = "are" if len(keys) > 1 else "is"
                added_codes.append(
                    (
                        [
                            "## Encoding Ordinal Categorical Features",
                            "Let's transfer categorical features as an integer array.",
                            "We will use Ordinal Encoder as explained [here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html).",
                            f"\nIn the given input dataset there {singular_verb} **{len(keys)}** column{singular} that can be transfered to integer and it includes: {recs}.",
                        ],
                        "markdown",
                    )
                )
        except Exception as ex:
            self.logger.warning(f"Column extraction issue ({ex})")
            return []
        return added_codes

    def __set_target__(self, loc, prev):
        added_codes = []
        try:
            tree = ast.parse(loc)
            target_col = [v.value for v in tree.body[0].value.elts]
            recs = ", ".join(target_col)
            singular = "s" if len(target_col) > 1 else ""
            singular_verb = "are" if len(target_col) > 1 else "is"
            added_codes.append(
                (
                    [
                        f"### Target Column{singular}",
                        f"We need to predict the target column{singular}.",
                        f"Therefore, we need to detach the target column{singular} in prediction.",
                        f"When the test data has the target column{singular}, the detaching is also executed for the test data."
                        f"Here {singular_verb} the list of *target column{singular}*: **{recs}**",
                    ],
                    "markdown",
                )
            )
        except Exception as ex:
            self.logger.warning(f"Column extraction issue ({ex})")
            return []
        return added_codes

    def __num_missing_data__(self, loc, prev):
        description = []
        keys = []
        tree = ast.parse(loc)
        if tree:
            if tree.body:
                if isinstance(tree.body[0], ast.Assign):
                    for key in tree.body[0].value.elts:
                        keys.append(key.value)
        recs = ", ".join(keys)
        singular = "s" if len(keys) > 1 else ""
        singular_verb = "are" if len(keys) > 1 else "is"
        description.append(
            (
                [
                    f"## Remove Missing Values in Numerical Column{singular}",
                    f"\nIn the given input dataset there {singular_verb} **{len(keys)} column{singular}** with missing data as follows: {recs}.",
                    f"\nThe following code removes the missing values from those column{singular}. We use an average value of each column or 0 to replace the Null values according to the missing value ratio.",
                ],
                "markdown",
            )
        )
        return description

    def __cat_missing_data__(self, loc, prev):
        description = []
        keys = []
        tree = ast.parse(loc)
        if tree:
            if tree.body:
                if isinstance(tree.body[0], ast.Assign):
                    for key in tree.body[0].value.elts:
                        keys.append(key.value)
        recs = ", ".join(keys)
        singular = "s" if len(keys) > 1 else ""
        singular_verb = "are" if len(keys) > 1 else "is"
        description.append(
            (
                [
                    f"## Remove Missing Values in Categorical Column{singular}",
                    f"\nIn the given input dataset there {singular_verb} **{len(keys)} column{singular}** with missing data as follows: {recs}.",
                    f"\nThe following code removes the missing values from those column{singular}. We use the most frequent value of each column or empty character to replace the Null values according to the missing value ratio.",
                ],
                "markdown",
            )
        )
        return description

    def __ordinal_encoder__(self, loc, prev):
        description = []
        keys = []
        tree = ast.parse(loc)
        recs = ", ".join(keys)
        singular = "s" if len(keys) > 1 else ""
        singular_verb = "are" if len(keys) > 1 else "is"
        if tree:
            if tree.body:
                if isinstance(tree.body[0], ast.Assign):
                    keys = [v.value for v in tree.body[0].value.elts]
                    description.append(
                        (
                            [
                                "## Encoding Ordinal Categorical Features",
                                "We will encode categorical features as an integer array.",
                                f"In the given input dataset there {singular_verb} **{len(keys)} column{singular}** with string values as follows: {recs}",
                            ],
                            "markdown",
                        )
                    )

        description.append((["The following code encode the selected columns."], "markdown"))
        return description

    def __num_to_scale__(self, loc, prev):
        description = []
        keys = []
        tree = ast.parse(loc)
        if tree:
            if tree.body:
                if isinstance(tree.body[0], ast.Assign):
                    keys = [v.value for v in tree.body[0].value.elts]
        recs = ", ".join(keys)
        singular = "s" if len(keys) > 1 else ""
        singular_verb = "are" if len(keys) > 1 else "is"
        description.append(
            (
                [
                    "## Numeric to Scale",
                    f"In the given input dataset there {singular_verb} **{len(keys)} column{singular}** with numeric values as follows where we can convert those values to scale through [log1p](https://numpy.org/doc/stable/reference/generated/numpy.log1p.html): {recs}.",
                ],
                "markdown",
            )
        )
        return description

    def __test_dataset_prediction_columns__exp__(self, loc, prev):
        added_codes = []
        try:
            if self.visualization_code:
                if "distplot" in self.visualization_code:
                    added_codes.append(
                        (
                            [
                                "## Visualization for data distribution of columns",
                            ],
                            "markdown",
                        )
                    )
                    added_codes.append((self.visualization_code["distplot"], "code"))
                if "heatmap" in self.visualization_code:
                    added_codes.append(
                        (
                            [
                                "## Visualization for feature heatmap",
                            ],
                            "markdown",
                        )
                    )
                    added_codes.append((self.visualization_code["heatmap"], "code"))
        except Exception as ex:
            self.logger.warning(f"Visualization processing issue ({ex})")
            return []
        return added_codes


class ExecuteNotebookThread(Thread):
    """ExecuteNotebookThread class."""

    def __init__(self, nb: NotebookNode, resources=None) -> None:
        self.ep = ExecutePreprocessor(timeout=6000, kernel_name="python3")
        self.nb = nb
        self.resources = resources
        self.exception = None
        Thread.__init__(self)

    def run(self):
        """run method."""
        try:
            self.ep.preprocess(self.nb, self.resources)
        except Exception as e:
            self.exception = e

    def trigger_interrupt_kernel(self):
        """trigger_interrupt_kernel method."""
        assert self.ep.km
        self.ep.km.interrupt_kernel()

    def get_exception(self):
        """get_exception method.
        Returns
        -------
        It returns the respective exception that occured.

        """
        return self.exception


class Miner:
    """Miner class."""

    def __init__(
        self,
        folder_path,
        init_blocks=[],
        visualization_code=None,
        logger=None,
        skeleton: Optional[dict] = None,
        explanation: Optional[dict] = None,
        run_info: Optional[dict] = None,
    ):
        """__init__ method.

        Parameters
        ----------
        folder_path : str
            Path of the folder.
        init_blocks : list[tuple]
            EDA description of corresponding blocks.
        visualization_code : AutoVisualization
            generated visualization code
        logger : None
            Logging Information.
        skeleton : dict, optional
            Probabilty score and other details of preprocess and model components.
        explanation : dict, optional
            pipelines explanation
        run_info : dict, optional
            execution results, logs and other information.

        """
        self.files = []
        self.content = []
        self.blocks = []
        self.templates = []
        self.visualization_code = visualization_code
        self.logger = logger
        if self.logger is None:
            self.logger = setup_logger()

        self.pipeline_explnation = None
        explains = None

        if skeleton is None:
            self.logger.warning("No Skeleton Information")
        if explanation is None:
            self.logger.warning("No Explanation Information")

        if skeleton and explanation:
            self.pipeline_explnation = Pipeline_Explanation(
                skeleton=skeleton, explanation=explanation, run_info=run_info
            )
            explains = self.pipeline_explnation.process()

        if os.path.isfile(folder_path) and folder_path.endswith("py"):
            # single file processing
            self.files.append(folder_path)
            dir_path = Path(folder_path).parent.absolute().as_posix()
        else:
            # batch processing
            self.files = glob(os.path.join(folder_path, "*.py"))
            dir_path = Path(folder_path).absolute().as_posix()

        self.output_path = dir_path
        self.local_path = dir_path

        for path in self.files:
            with open(path, "r", encoding="utf-8") as f:
                code = f.read()

            code = "\n".join(code.split("\n")[2:])

            current_block = self.get_block(code)
            self.blocks.append(current_block)
            updated_block = self.add_template_block(current_block, self.visualization_code)
            self.content.append(
                {
                    "path": path,
                    "filename": "".join(path.split("/")[-1].split(".")[:-1]),
                    "code": code,
                    "blocks": updated_block,
                }
            )

        self.templates = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 4}

        for index, rec in enumerate(self.content):
            jnote = self.templates
            for block in init_blocks:
                # adding initial_blocks if provided (i.e., from AutoEDA)
                cell_rows, cell_type = block
                if cell_type == "markdown":
                    cell = {"cell_type": "markdown", "metadata": {}, "source": "\n".join(cell_rows)}
                else:
                    # process code type cells
                    cell = {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": "\n".join(cell_rows),
                    }
                jnote["cells"].append(cell)

            for block in rec["blocks"]:
                # processing generated blocks
                cell_rows, cell_type = block
                if cell_type == "markdown":
                    cell = {"cell_type": "markdown", "metadata": {}, "source": "\n".join(cell_rows)}
                else:
                    if explains:
                        # code cell
                        # TODO: optimize it or use a tag
                        explain_cell = None
                        found_counter = 0
                        for block_loc in block[0]:
                            if block_loc.strip():
                                for iexplain, explain in enumerate(explains):
                                    if not explain["added"]:
                                        for ex_code in explain["code"]:
                                            for ex_loc in ex_code.split("\n"):
                                                if (
                                                    not (ex_loc.startswith("from ") or ex_loc.startswith("import "))
                                                    and ex_loc.strip()
                                                    and ex_loc in block_loc
                                                ):
                                                    if found_counter == 0:
                                                        found_counter += 1  # at least two matched loc needed
                                                    else:
                                                        explain_cell = {
                                                            "cell_type": "markdown",
                                                            "metadata": {},
                                                            "source": explain["explanation"],
                                                        }
                                                        explains[iexplain]["added"] = True
                                                        found_counter = 0
                                                        break
                                    if explain_cell:
                                        break
                                if explain_cell:
                                    break
                        if explain_cell:
                            jnote["cells"].append(explain_cell)

                    cell = {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": "\n".join(cell_rows),
                    }

                jnote["cells"].append(cell)
            rec["jupyter"] = jnote

    @staticmethod
    def __save__(filename, notebook_object):
        with open(filename, "w", encoding="utf-8") as fw:
            json.dump(notebook_object, fw)

    @staticmethod
    def execute_notebook(
        nb: NotebookNode, resources=None, timeout: int = 0, cancel: Optional[CancellationToken] = None
    ):
        """execute_notebook method.

        Parameters
        ----------
        nb : NotebookNode
            Jupyter Notebook
        resources : None
            Resources needed to run the notebook.
        timeout : int
            Integer value for timeout.
        cancel : CancellationToken, optional

        """
        ep_thread = ExecuteNotebookThread(nb, resources)
        ep_thread.start()

        start_time = time.time()

        is_interrupted = False
        while not is_interrupted and ep_thread.is_alive():
            if (timeout > 0) and (time.time() - start_time) > timeout:
                ep_thread.trigger_interrupt_kernel()
                is_interrupted = True
            if cancel and cancel.is_triggered:
                ep_thread.trigger_interrupt_kernel()
                is_interrupted = True
            time.sleep(1)
        ep_thread.join()
        exception = ep_thread.get_exception()
        if exception:
            raise exception

    def save_all(self, execution=False, timeout: int = 0, cancel: Optional[CancellationToken] = None):
        """save_all method.

        Parameters
        ----------
        execution : bool
            False and otherwise True.
        timeout : int
            Integer value for timeout.
        cancel : CancellationToken, optional

        Returns
        -------
        output_files : list[str]
            List of saved explained jupyter notebooks..

        """
        output_files = []
        for rec in self.content:
            path = os.path.join(self.output_path, rec["filename"] + ".ipynb")
            self.__save__(path, rec["jupyter"])
            self.logger.info(f"saved:{path}")
            if execution and (cancel is None or cancel.is_triggered is False):
                try:
                    self.logger.info("Running the explained notebook...")
                    with open(path, "r", encoding="utf-8") as f:
                        nb = nbformat.read(f, as_version=4)
                    self.execute_notebook(nb, {"metadata": {"path": self.output_path}}, timeout, cancel)
                except Exception:
                    import traceback

                    self.logger.warning(f"Failed to execute notebook: {path}.ipynb, error: {traceback.format_exc()}")
                finally:
                    with open(f"{path}.out.ipynb", "w", encoding="utf-8") as f:
                        nbformat.write(nb, f)
                    output_files.append(f"{path}.out.ipynb")
                    self.logger.info(f"Saved explained notebook in: {path}.out.ipynb")
        return output_files

    def get_block(self, code):
        """get_block method.

        generate a set of blocks from LoCs
        each block seperated through empty lines (\n)

        Parameters
        ----------
        code : str
            code from the python files.

        Returns
        -------
        blocks : list[str]
            set of blocks from LoCs.
            each block seperated through empty lines (\n)

        """
        blocks = []
        current_block = []
        eob = False
        for line in code.split("\n"):
            if len(line.strip()) == 0:
                eob = True
            else:
                if (line.startswith("\t") or line.startswith(" ")) and eob:
                    eob = False
                if (line.startswith("#") or line.startswith("'''")) and eob:
                    # generate a new block
                    blocks.append(current_block)
                    current_block = []
                    eob = False
                current_block.append(line)
        if len(blocks):
            blocks.append(current_block)
        return blocks

    def add_template_block(
        self, blocks, visualization_code, content_template_path=os.path.join("templates", "jupyter_content.json")
    ):
        """add_template_block method.

        Parameters
        ----------
        blocks : list[str]
            set of blocks from LoCs.
            each block seperated through empty lines (\n)
        visualization_code : AutoVisualization
            generated visualization code.
        content_template_path : Path
            Path to the jupyter content templates.

        Returns
        -------
        final_block : list[tuples]
            Adding markdown cells before and after blocks in code from jupyter content template.

        """
        ct = Code_Template()
        desc = AST_Update(visualization_code, local_path=self.local_path)
        final_block = []
        try:
            path = Path(__file__).parent / content_template_path
            jupyter_content_template = json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            self.logger.error(f"Template file not found:{content_template_path}")
            raise
        prev = ""
        for block in blocks:
            before_block = []
            after_block = []
            extended_lines = []
            for line in block:
                if line.strip() in jupyter_content_template:
                    before_block, after_block = jupyter_content_template[line.strip()]
                try:
                    desc_lines = desc.process(line, prev)
                    if desc_lines:
                        extended_lines.extend(desc_lines)
                except NotImplementedError:
                    raise
                prev = line

            if before_block:
                before_block = ct.update(before_block)
                final_block.append((before_block, "markdown"))
            if extended_lines:
                final_block.extend(extended_lines)

            final_block.append((block, "code"))

            if after_block:
                after_block = ct.update(after_block)
                final_block.append((after_block, "markdown"))

        return final_block
