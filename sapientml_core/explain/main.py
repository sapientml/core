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

from typing import Literal, Optional

import pandas as pd
from sapientml.params import CancellationToken
from sapientml.util.logging import setup_logger
from sapientml_core.preprocess.default.generator import check_cols_has_symbols, remove_symbols, rename_cols

from .AutoEDA import EDA
from .AutoVisualization import AutoVisualization_Class
from .code_miner import Miner

logger = setup_logger()


def process(
    visualization: bool,
    eda: bool,
    dataframe: pd.DataFrame,
    script_path: str,
    target_columns: list[str],
    problem_type: Literal["regression", "classification"],
    ignore_columns: Optional[list[str]] = None,
    skeleton: Optional[dict] = None,
    explanation: Optional[dict] = None,
    run_info: Optional[dict] = None,
    internal_execution: bool = False,
    timeout: int = 0,
    cancel: Optional[CancellationToken] = None,
):
    """process function.

    Parameters
    ----------
    visualization : bool
        True and otherwise False
    eda : bool
        True and otherwise False
    dataframe : pd.DataFrame
        dataframe input
    script_path : str
        Path of the script.
    target_columns : list[str]
        Names of target columns.
    problem_type : Literal["regression", "classification"]
        Type of problem either regression or classification
    ignore_columns : list[str], optional
        Column names which must not be used and must be dropped.
    skeleton : dict, optional
        Probabilty score and other details of preprocess and model components.
    explanation : dict, optional
        pipelines explanation
    run_info : dict, optional
        execution results, logs and other information.
    internal_execution : bool
        True and otherwise Flase
    timeout : int
        integer value for timeout
    cancel : CancellationToken, optional

    Returns
    -------
    output_files : List[str]
        list of .ipynb files.

    """
    output_files = None

    if visualization:
        # Call AutoVisualization to generate visualization codes
        AV = AutoVisualization_Class()
        cols_has_symbols = check_cols_has_symbols(dataframe.columns.to_list())
        no_symbol_columns = [col for col in dataframe.columns.values if col not in cols_has_symbols]
        if cols_has_symbols:
            rename_dict = {}
            org_df_column = dataframe.columns.to_list()
            df_columns = list(
                dataframe.rename(columns=lambda col: remove_symbols(col) if col in cols_has_symbols else col).columns
            )
            rename_dict = rename_cols(org_df_column, no_symbol_columns, df_columns)
            if len(rename_dict) != 0:
                col_has_target = []
                for org_column, target in zip(list(rename_dict.keys()), list(rename_dict.values())):
                    if target in target_columns:
                        col_has_target.append(org_column)
                visualization_code = AV.AutoVisualization(
                    df=dataframe,
                    target_columns=col_has_target,
                    problem_type=problem_type,
                    ignore_columns=ignore_columns,
                )
            else:
                visualization_code = AV.AutoVisualization(
                    df=dataframe,
                    target_columns=col_has_target,
                    problem_type=problem_type,
                    ignore_columns=ignore_columns,
                )
        else:
            visualization_code = AV.AutoVisualization(
                df=dataframe,
                target_columns=target_columns,
                problem_type=problem_type,
                ignore_columns=ignore_columns,
            )
    else:
        visualization_code = None

    if eda:
        # handle list(tuple, dict) value in dataframe.
        for col in dataframe.columns:
            exist_list_values = [x for x in dataframe[col] if type(x) in [list, tuple, dict]]
            if len(exist_list_values) > 0:
                dataframe[col] = dataframe[col].fillna("").astype(str)
        eda = EDA(dataframe, target_columns, log_level=2)

        eda.check_consistency(convert=False)

        categories, desc = eda.cat_process(threshold=0.01, IQR_activation=True, z_activation=True)

        initial_blocks = eda.description
    else:
        initial_blocks = []

    code_miner = Miner(
        script_path,
        init_blocks=initial_blocks,
        visualization_code=visualization_code,
        logger=logger,
        skeleton=skeleton,
        explanation=explanation,
        run_info=run_info,
    )
    output_files = code_miner.save_all(execution=internal_execution, timeout=timeout, cancel=cancel)
    return output_files
