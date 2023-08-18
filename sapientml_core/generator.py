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

import ast
import copy
import re
from importlib.metadata import entry_points
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sapientml.generator import CodeBlockGenerator, PipelineGenerator
from sapientml.params import CancellationToken, Code, Config, Dataset, PipelineResult, RunningResult, Task
from sapientml.result import SapientMLGeneratorResult
from sapientml.util.logging import setup_logger

from . import ps_macros
from .adaptation.generation.template_based_adaptation import Adaptation
from .explain.main import process as explain
from .params import Pipeline, summarize_dataset
from .seeding.predictor import predict

model_dir_path_default = Path(__file__).parent / "models"

logger = setup_logger()


def _is_strnum_column(c):
    c2 = c.loc[c.notnull()]
    c2 = pd.to_numeric(c2, errors="coerce")
    ratio = c2.notnull().sum() / c2.shape[0]
    return ratio > 0.9


class SapientMLGenerator(PipelineGenerator, CodeBlockGenerator):
    def __init__(self, config: Config):
        CodeBlockGenerator.__init__(self, config)
        eps = entry_points(group="code_block_generator")
        self.loaddata = eps["loaddata"].load()(config)
        self.preprocess = eps["preprocess"].load()(config)

    def generate_pipeline(self, dataset: Dataset, task: Task) -> list[Code]:
        dataset, loaddata_block = self.loaddata.generate_code(dataset, task)
        dataset, preprocess_block = self.preprocess.generate_code(dataset, task)
        code_block = loaddata_block + preprocess_block
        dataset, sapientml_results = self.generate_code(dataset, task)

        result_pipelines: list[Code] = []
        for pipeline in sapientml_results:
            pipeline.validation = code_block.validation + pipeline.validation
            pipeline.test = code_block.test + pipeline.test
            pipeline.train = code_block.train + pipeline.train
            pipeline.predict = code_block.predict + pipeline.predict
            result_pipelines.append(pipeline)
        return result_pipelines

    def generate_code(self, dataset: Dataset, task: Task) -> Tuple[Dataset, list[Pipeline]]:
        df = dataset.training_dataframe
        dataset_summary = summarize_dataset(df, task)  # type: ignore
        if dataset_summary.has_inf_value_targets:
            raise ValueError("Stopped generation because target columns have infinity value.")

        # discard columns with analysis
        # NOTE: The following code modify task.ignore_columns because ignore_columns is the same instance as task.ignore_columns.
        # 1. columns marked as STR_OTHER
        if ps_macros.STR_OTHER in dataset_summary.meta_features_pp:
            undetermined_column_names = dataset_summary.meta_features_pp[ps_macros.STR_OTHER]
            if isinstance(undetermined_column_names, list):
                task.ignore_columns += undetermined_column_names
        del dataset_summary.meta_features_pp[ps_macros.STR_OTHER]
        # 2. columns with all null values
        if ps_macros.ALL_MISSING_PRESENCE in dataset_summary.meta_features_pp:
            column_names_with_all_missing_values = dataset_summary.meta_features_pp[ps_macros.ALL_MISSING_PRESENCE]
            if isinstance(column_names_with_all_missing_values, list):
                task.ignore_columns += column_names_with_all_missing_values
        del dataset_summary.meta_features_pp[ps_macros.ALL_MISSING_PRESENCE]

        labels = predict(task, dataset_summary)
        adapt = Adaptation(
            labels,
            task,
            dataset_summary,
            self.config,
        )
        pipelines = adapt.run_adaptation()

        return dataset, pipelines

    def evaluate(self, pipeline_results: list[tuple[Code, RunningResult]], lower_is_better: bool = False) -> None:
        self._best_pipeline = None
        self._best_pipeline_score = PipelineResult(score=None, metric=None, best_params=None)
        candidate_scripts = []
        for pipeline, result in pipeline_results:
            if result.returncode == 0:
                pipeline_score = self._parse_pipeline_output(result.output)
            else:
                pipeline_score = PipelineResult(score=None, metric=None, best_params=None)
            candidate_scripts.append((pipeline, pipeline_score))
        self._candidate_scripts = candidate_scripts

        # When an error occurs while running a pipeline, the score becomes None
        error_pipelines = [pipeline for pipeline in candidate_scripts if pipeline[1].score is None]

        # If none of them have the score, stop ranking them
        if len(candidate_scripts) == len(error_pipelines):
            return

        # sort descending
        succeeded_scripts = sorted(
            [x for x in candidate_scripts if x[1].score is not None],
            key=lambda x: x[1].score,
            reverse=(not lower_is_better),
        )
        failed_scripts = [x for x in candidate_scripts if x[1].score is None]

        ranked_candidate_scripts = succeeded_scripts + failed_scripts
        best_pipeline_tuple = ranked_candidate_scripts[0]
        if best_pipeline_tuple is None:
            return

        best_pipeline = copy.deepcopy(best_pipeline_tuple[0])
        if best_pipeline_tuple[1].best_params is not None:
            best_pipeline.test = best_pipeline.test.replace(
                "best_params = study.best_params", "best_params = " + str(best_pipeline_tuple[1].best_params)
            )
            best_pipeline.train = best_pipeline.train.replace(
                "best_params = study.best_params", "best_params = " + str(best_pipeline_tuple[1].best_params)
            )
        self._best_pipeline = best_pipeline
        self._best_pipeline_score = best_pipeline_tuple[1]


    def get_result(self):
        return (self._best_pipeline, self._best_pipeline_score), self._candidate_scripts

    @staticmethod
    def _parse_pipeline_output(output: str):
        score = None
        best_params = None
        metric = None
        output_lines = output.splitlines()
        try:
            for line in output_lines:
                if re.match("best params: ", line):
                    best_params = ast.literal_eval(re.findall("best params: (.+)", line)[0])
                elif re.match("RESULT: ", line):
                    parts = [x.strip() for x in line.split(":")]
                    metric = parts[-2].strip().split(" ")[0]
                    score = float(parts[-1])
        except Exception:
            pass
        return PipelineResult(score=score, metric=metric, best_params=best_params)

    def save(
        self,
        result: SapientMLGeneratorResult,
        output_dir_path: str,
        project_name: str = "",
        cancel: Optional[CancellationToken] = None,
    ):
        if self._best_pipeline is None:
            return

        skeleton = self._best_pipeline.labels

        def add_prefix(filename, prefix):
            if not prefix:
                return filename
            return f"{prefix}_{filename}"

        debug_info = {}
        for i, candidate in enumerate(self._candidate_scripts):
            info = {"content": candidate[0].dict(), "run_info": candidate[1].__dict__}
            debug_info[i] = info

        explain(
            visualization=True,
            eda=True,
            dataframe=result.training_data,
            script_path=(Path(output_dir_path) / add_prefix("final_script.py", project_name)).absolute().as_posix(),
            target_columns=result.target_columns,
            problem_type=result.task_type,
            ignore_columns=result.ignore_columns,
            skeleton=skeleton,
            explanation=self._best_pipeline.pipeline_json,
            run_info=debug_info,
            internal_execution=True,
            timeout=result.timeout_for_test,
            cancel=cancel,
        )
