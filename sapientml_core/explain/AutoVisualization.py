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

import warnings
from collections import defaultdict
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sapientml.util.logging import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger()


class AutoVisualization_Class:
    """AutoVisualization_Class."""

    def __init__(self):
        self.code_json = defaultdict(list)

    def AutoVisualization(
        self,
        df: pd.DataFrame,
        target_columns: list[str],
        problem_type: Literal["regression", "classification"],
        ignore_columns: Optional[list[str]] = None,
    ):
        """
        The main processing class for visualizations.

        Parameters
        ----------
        target_columns : list[str]
            Names of target columns.
        problem_type : Literal["regression", "classification"]
            Type of problem either regression or classification
        ignore_columns : list[str], optional
            Column names which must not be used and must be dropped.

        Returns
        -------
        generates visualization code.

        """
        problem_type = problem_type.lower()

        if not ignore_columns:
            ignore_columns = []

        if problem_type != "regression" and problem_type != "classification":
            logger.info(
                "Problem type should be configured either regression or classification. Please check your task specification"
            )
            # if problem type is not specified as either 'regression' or 'classification', set it to default value 'regression'
            problem_type = "regression"

        if not isinstance(target_columns, list):
            logger.error("Target columns should be defined as a list. Please check your task specification")
            return self.code_json

        if len(target_columns) == 0:
            logger.error("Target column is not configured. Please check your task specification")
            return self.code_json

        if not isinstance(ignore_columns, list):
            logger.info("ID columns should be defined as a list. Please check your task specification")

        numvars_list, catvars_list, textvars_list = self.classify_columns_coarse_granularity(
            df, ignore_columns, target_columns
        )

        numvars_list = sorted(numvars_list)
        catvars_list = sorted(catvars_list)
        textvars_list = sorted(textvars_list)
        target_columns = sorted(target_columns)

        try:
            self.draw_distplot(problem_type, numvars_list, catvars_list, target_columns)
        except Exception:
            logger.error("Could not draw Distribution Plots")

        try:
            self.draw_heatmap(problem_type, numvars_list, catvars_list, target_columns)
        except Exception:
            logger.error("Could not draw Heat Maps")

        logger.info("VISUALIZATION Completed.")

        return self.code_json

    def draw_heatmap(self, problem_type, numvars_list=None, catvars_list=None, targetvars_list=None):
        """draw_heatmap method.

        generate codes for feature heatmap.

        Parameters
        ----------
        problem_type : Literal["regression", "classification"]
            Type of problem either regression or classification
        numvars_list : List, default None
            list of numerical columns
        catvars_list : List, default None
            list of categorical columns
        targetvars_list : List, default None
            list of target columns.

        """
        total_numvars = len(numvars_list)
        total_catvars = len(catvars_list)

        max_vis_to_show = 20  # set a upper limits

        heatmap_list = (numvars_list + catvars_list)[:max_vis_to_show]
        heatmap_list.extend(targetvars_list)

        if total_numvars + total_catvars > 1:
            # generate the codes
            codes = [
                "_COLS_FOR_HEATMAP = %s" % str(heatmap_list),
                "plt.figure(figsize=(20,20))",
                'plt.title("Pearson Correlation HeatMap of Features")',
                "sns.heatmap(train_dataset[_COLS_FOR_HEATMAP].corr(numeric_only=True),annot=True)",
            ]

            self.code_json["heatmap"] = codes

    def draw_distplot(self, problem_type, numvars_list=None, catvars_list=None, targetvars_list=None):
        """
        draw_distplot method.

        generate codes for data distribution plots.

        Parameters
        ----------
        problem_type : Literal["regression", "classification"]
            Type of problem either regression or classification
        numvars_list : List | None, default None
            list of numerical columns.
        catvars_list : List | None, default None
            list of categorical columns.
        targetvars_list : List | None, default None
            list of target columns.

        """
        total_numvars = len(numvars_list)
        total_catvars = len(catvars_list)
        total_targetvars = len(targetvars_list)

        max_vis_to_show = 100  # set a upper limits

        if total_numvars >= max_vis_to_show:
            numvars_list = numvars_list[:max_vis_to_show]
            total_numvars = 100
        if total_catvars >= max_vis_to_show:
            catvars_list = catvars_list[:max_vis_to_show]
            total_catvars = 100
        if total_targetvars >= max_vis_to_show:
            targetvars_list = targetvars_list[:max_vis_to_show]
            total_targetvars = 100

        # best layout for numerical columns
        numvars_n_row, numvars_n_col, numvars_n_unvisible = 0, 0, 0
        if total_numvars <= 4:
            numvars_n_row, numvars_n_col = 1, total_numvars
        else:
            numvars_n_row, numvars_n_col = total_numvars // 4, 4
            if total_numvars % 4 != 0:
                numvars_n_row += 1  # need one more row
                numvars_n_unvisible = 4 - total_numvars % 4

        # best layout for categorical columns
        catvars_n_row, catvars_n_col, catvars_n_unvisible = 0, 0, 0
        if total_catvars <= 4:
            catvars_n_row, catvars_n_col = 1, total_catvars
        else:
            catvars_n_row, catvars_n_col = total_catvars // 4, 4
            if total_catvars % 4 != 0:
                catvars_n_row += 1  # need one more row
                catvars_n_unvisible = 4 - total_catvars % 4

        # best layout for target columns
        targetvars_n_row, targetvars_n_col, targetvars_n_unvisible = 0, 0, 0
        if total_targetvars <= 4:
            targetvars_n_row, targetvars_n_col = 1, total_targetvars
        else:
            targetvars_n_row, targetvars_n_col = total_targetvars // 4, 4
            if total_targetvars % 4 != 0:
                targetvars_n_row += 1  # need one more row
                targetvars_n_unvisible = 4 - total_targetvars % 4

        # generate the codes
        codes = [
            "import matplotlib.pyplot as plt",
            "import japanize_matplotlib",
            "import seaborn as sns",
            "import numpy as np",
            "\n",
        ]

        # add numerical columns related codes
        if total_numvars > 1:
            codes.extend(
                [
                    "fig, axes = plt.subplots(%d, %d, figsize=(%d, %d))"
                    % (numvars_n_row, numvars_n_col, 16, numvars_n_row * 4),
                    "axes = axes.ravel()",
                    "for index,col in enumerate(" + str(numvars_list) + "):",
                    "\tsns.histplot(train_dataset[col],ax=axes[index])",
                    '\taxes[index].tick_params(axis="x", rotation=90)',
                    '\taxes[index].set_title("Distribution of %s" %col )',
                    "\tif len(axes[index].get_xticks()) > 20:",
                    "\t\txtick_hidden = len(axes[index].get_xticks())//20",
                    "\t\taxes[index].set_xticks(axes[index].get_xticks()[::xtick_hidden])",
                ]
            )

            if numvars_n_unvisible > 1:
                codes.extend(
                    [
                        "for index in range(%d,0):" % (-numvars_n_unvisible),
                        "\taxes[index].set_visible(False)",
                        "fig.tight_layout()",
                        "\n",
                    ]
                )
            elif numvars_n_unvisible == 1:
                codes.extend(["axes[-1].set_visible(False)", "fig.tight_layout()", "\n"])
            else:
                codes.extend(["fig.tight_layout()", "\n"])

        if total_numvars == 1:
            codes.extend(
                [
                    "fig = plt.figure(figsize=(4,4))",
                    'sns.histplot(x = train_dataset["' + str(numvars_list[0]) + '"])',
                    "plt.xticks(rotation=90)",
                    "plt.locator_params(nbins=20)",
                    'plt.title("Distribution of %s")' % str(numvars_list[0]),
                    "fig.tight_layout()" "\n",
                ]
            )

        # add categorical columns related codes
        if total_catvars > 1:
            codes.extend(
                [
                    "fig, axes = plt.subplots(%d, %d, figsize=(%d, %d))"
                    % (catvars_n_row, catvars_n_col, 16, catvars_n_row * 4),
                    "axes = axes.ravel()",
                    "for index,col in enumerate(" + str(catvars_list) + "):",
                    '\tsns.countplot(x = train_dataset[col].fillna("").astype(str), ax=axes[index], order=sorted(train_dataset[col].fillna("").astype(str).unique()))',
                    '\taxes[index].tick_params(axis="x", rotation=90)',
                    '\taxes[index].set_title("Distribution of %s" %col )',
                    "\tif len(axes[index].get_xticks()) > 20:",
                    "\t\txtick_hidden = len(axes[index].get_xticks())//20",
                    "\t\taxes[index].set_xticks(axes[index].get_xticks()[::xtick_hidden])",
                ]
            )

            if catvars_n_unvisible > 1:
                codes.extend(
                    [
                        "for index in range(%d,0):" % (-catvars_n_unvisible),
                        "\taxes[index].set_visible(False)",
                        "fig.tight_layout()",
                        "\n",
                    ]
                )
            elif catvars_n_unvisible == 1:
                codes.extend(["axes[-1].set_visible(False)", "fig.tight_layout()", "\n"])
            else:
                codes.extend(["fig.tight_layout()", "\n"])

        if total_catvars == 1:
            codes.extend(
                [
                    "fig = plt.figure(figsize=(4,4))",
                    'sns.countplot(x = train_dataset["'
                    + str(catvars_list[0])
                    + '"].fillna("").astype(str), order=sorted(train_dataset["'
                    + str(catvars_list[0])
                    + '"].fillna("").astype(str).unique()))',
                    "plt.xticks(rotation=90)",
                    "plt.locator_params(nbins=20)",
                    'plt.title("Distribution of %s")' % str(catvars_list[0]),
                    "fig.tight_layout()" "\n",
                ]
            )

        # add target columns related codes
        if total_targetvars > 1:
            codes.extend(
                [
                    "fig, axes = plt.subplots(%d, %d, figsize=(%d, %d))"
                    % (targetvars_n_row, targetvars_n_col, 16, targetvars_n_row * 4),
                    "axes = axes.ravel()",
                    "for index,col in enumerate(" + str(targetvars_list) + "):",
                ]
            )

            if problem_type == "regression":
                codes.extend(
                    [
                        "\tsns.histplot(x = train_dataset[col],ax=axes[index])",
                        '\taxes[index].tick_params(axis="x", rotation=90)',
                        '\taxes[index].set_title("Distribution for Target Column of %s" %col )',
                    ]
                )
            else:
                # NOTE: In version 2.6.9, multi-target classification doesn't working.
                #       So, this route can't be tested.
                codes.extend(
                    [
                        '\tsns.countplot(x = train_dataset[col].fillna("").astype(str), ax=axes[index], order=sorted(train_dataset[col].fillna("").astype(str).unique()))',
                        '\taxes[index].tick_params(axis="x", rotation=90)',
                        '\taxes[index].set_title("Distribution for Target Column of %s" %col )',
                    ]
                )

            if targetvars_n_unvisible > 1:
                codes.extend(
                    [
                        "for index in range(%d,0):" % (-targetvars_n_unvisible),
                        "\taxes[index].set_visible(False)",
                        "fig.tight_layout()",
                        "\n",
                    ]
                )
            elif targetvars_n_unvisible == 1:
                codes.extend(["axes[-1].set_visible(False)", "fig.tight_layout()", "\n"])
            else:
                codes.extend(["fig.tight_layout()", "\n"])

        if total_targetvars == 1:
            codes.extend(
                [
                    "fig = plt.figure(figsize=(5,5))",
                ]
            )

            if problem_type == "regression":
                codes.extend(
                    [
                        'sns.histplot(x = train_dataset["' + str(targetvars_list[0]) + '"])',
                    ]
                )
            else:
                codes.extend(
                    [
                        'sns.countplot(x = train_dataset["'
                        + str(targetvars_list[0])
                        + '"].fillna("").astype(str), order=sorted(train_dataset["'
                        + str(targetvars_list[0])
                        + '"].fillna("").astype(str).unique()))',
                    ]
                )

            codes.extend(
                [
                    "plt.xticks(rotation=90)",
                    'plt.title("Distribution for Target Column of %s")' % str(targetvars_list[0]),
                    "fig.tight_layout()" "\n",
                ]
            )

        self.code_json["distplot"] = codes

    def classify_columns_coarse_granularity(self, df_preds, ignore_columns, target_columns, verbose=3):
        """
        classify_columns_coarse_granularity method.

        simple classification function for columns.
        only to classify the columns into numerical, category and text columns.

        Parameters
        ----------
        df_preds : pd.DataFrame
            input dataframe
        ignore_columns : list[str], optional
            Column names which must not be used and must be dropped.
        target_columns : list[str]
            Names of target columns.

        Results
        -------
        numvars_list : List
            list of numerical columns
        catvars_list : List
            list of categorical columns
        textvars_list : List
            list of text columns.

        """
        # categorical column threshold
        cat_threshold = 0.05
        catvars_list = []
        numvars_list = []
        textvars_list = []

        for var in df_preds.columns:
            # add .astype(str) to avoid error if dataframe has list values when caluculate nunique.
            if 1.0 * df_preds[var].fillna("").astype(str).nunique() / df_preds[var].count() < cat_threshold:
                if (
                    var not in ignore_columns and var not in target_columns
                ):  # id column and target column need to be removed from consideration
                    catvars_list.append(var)

        object_vars_all = df_preds.select_dtypes(include="object").columns.tolist()
        num_vars_all = df_preds.select_dtypes(include=np.number).columns.tolist()

        for var in num_vars_all:
            if var not in catvars_list:
                if (
                    var not in ignore_columns and var not in target_columns
                ):  # id column and target column need to be removed from consideration
                    numvars_list.append(var)

        for var in object_vars_all:
            if var not in catvars_list:
                if (
                    var not in ignore_columns and var not in target_columns
                ):  # id column and target column need to be removed from consideration
                    textvars_list.append(var)

        return numvars_list, catvars_list, textvars_list
