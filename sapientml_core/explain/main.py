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

from typing import Literal, Optional

import pandas as pd
from sapientml.params import CancellationToken
from sapientml.util.logging import setup_logger

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
