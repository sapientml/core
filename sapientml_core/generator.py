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
import copy
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from glob import glob
from pathlib import Path
from shutil import copyfile
from typing import Tuple, Union

from sapientml.executor import PipelineExecutor
from sapientml.generator import CodeBlockGenerator, PipelineGenerator
from sapientml.macros import metric_lower_is_better
from sapientml.params import Code, Dataset, PipelineResult, RunningResult, Task
from sapientml.util.json_util import JSONEncoder
from sapientml.util.logging import setup_logger
from tqdm import tqdm

from . import internal_path
from .adaptation.generation.template_based_adaptation import Adaptation
from .explain.main import process as explain
from .params import SapientMLConfig, SimplePipeline, summarize_dataset
from .seeding.predictor import predict
from .training import project_corpus

if sys.version_info.minor <= 9:
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

logger = setup_logger()


def add_prefix(filename, prefix):
    """Add prefix to filename if prefix exists.

    Parameters
    ----------
    filename : str
        Filename.
    prefix : str
        Prefix(project_name).

    Returns
    ----------
    str
        If prefix exists, return f"{prefix}_{filename}",
        If prefix does not exist, return only filename.
    """
    if not prefix:
        return filename
    return f"{prefix}_{filename}"


class SapientMLGenerator(PipelineGenerator, CodeBlockGenerator):
    """
    Own the main functions for generating the pipeline.
    """

    def __init__(self, datastore="localfile", preprocess="default", **kwargs):
        self.config = SapientMLConfig(**kwargs)
        self.config.post_init()
        eps_datastore = entry_points(group="sapientml.datastore")
        eps_preprocess = entry_points(group="sapientml.preprocess")
        self.datastore = eps_datastore[datastore].load()(**kwargs)
        self.preprocess = eps_preprocess[preprocess].load()(**kwargs)

    def train(self, tag=None, num_parallelization=200):
        """Run meta-training.

        Parameters
        ----------
        tag : str | None
            The tag of souce code.
            If tag is set, the traning results will be saved into <core/sapientml_core/.cache/tag/>.
            Else if tag is not set, the traning results will be saved into <core/sapientml_core/.cache/>.
        num_parallelization : int | 200
            Number of parallelization.
            Default value is 200.
        """

        def _exec(cmd):
            logger.info(f"Executing {cmd}")
            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = proc.communicate()
            logger.debug("STDOUT: {}".format(stdout))
            logger.error("STDERR:{}".format(stderr))
            return proc

        def _wait_for(file_pattern, n_files=1):
            logger.info(f"Waiting for {file_pattern}" + (f" * {n_files}" if n_files > 1 else ""))
            latest = 0
            with tqdm(total=n_files) as pbar:
                pbar.update(latest)
                while latest < n_files:
                    time.sleep(2)
                    k = len(glob(f"{internal_path.training_cache}/{file_pattern}"))
                    if k > latest:
                        pbar.update(k - latest)
                        latest = k

        corpus = project_corpus.ProjectCorpus()
        projects = corpus.project_list

        if tag:
            internal_path.training_cache = internal_path.training_cache / tag

        os.makedirs(internal_path.training_cache, exist_ok=True)

        start = datetime.now()

        logger.info(f"Start: {start}")

        # Step-1
        _exec(
            f"PYTHONPATH=. python {internal_path.sapientml_core_root}/training/denoising/static_analysis_of_columns.py --tag={tag}"
        )
        _wait_for("static_info.json")
        _exec(
            f"PYTHONPATH=. python {internal_path.sapientml_core_root}/training/denoising/dataset_snapshot_extractor.py --tag={tag}"
        )
        _wait_for("dataset-snapshots/*.txt", len(projects))

        # Step-2
        procs = []
        for x in range(num_parallelization):
            proc = _exec(
                f"PYTHONPATH=. python {internal_path.sapientml_core_root}/training/augmentation/mutation_runner.py {num_parallelization} {x} {tag}"
            )
            procs.append(proc)

        _wait_for("exec_info/mutation_*.finished", num_parallelization)

        for p in procs:
            if p.poll() is None:
                logger.info("Kill alive processes")
                p.terminate()
                logger.info("Done, exit status:{}".format(proc.poll()))

        _exec(
            f"PYTHONPATH=. python {internal_path.sapientml_core_root}/training/denoising/determine_used_features.py --tag={tag}"
        )
        _wait_for("feature_analysis_summary.json")
        _exec(
            f"PYTHONPATH=. python {internal_path.sapientml_core_root}/training/augmentation/mutation_results.py --tag={tag}"
        )

        # Step-3
        _exec(f"PYTHONPATH=. python {internal_path.sapientml_core_root}/training/meta_feature_extractor.py --tag={tag}")
        _wait_for("*_metafeatures_training.csv", 2)

        # Step-4
        _exec(f"PYTHONPATH=. python {internal_path.sapientml_core_root}/training/pp_model_trainer.py --tag={tag}")
        _exec(f"PYTHONPATH=. python {internal_path.sapientml_core_root}/training/meta_model_trainer.py --tag={tag}")

        _wait_for("*.pkl", 3)

        # Step-5
        _exec(
            f"PYTHONPATH=. python {internal_path.sapientml_core_root}/training/dataflowmodel/dependent_api_extractor.py --tag={tag}"
        )
        _wait_for("dependent_labels.json")
        _exec(
            f"PYTHONPATH=. python {internal_path.sapientml_core_root}/training/dataflowmodel/determine_label_order.py --tag={tag}"
        )
        _wait_for("label_order.json")

        end = datetime.now()
        logger.info(f"End: {end}")

        diff_time = end - start
        logger.info(f"Training time cost: {diff_time}")

    def generate_pipeline(self, dataset: Dataset, task: Task):
        """Generate candidate scripts, execute them and select the best one.

        Parameters
        ----------
        dataset : Dataset
            Dataset class.
        task : Task
            Task class.

        Returns
        ----------
        (best_pipeline, best_pipeline_score) : Tuple(list | None, PipelineResult)
            best_pipeline: Best pipeline.
            best_pipline_score: The score of the best pipeline.
        candidate_scripts : list
            Candidate scripts with scores.
        """
        self.dataset = dataset
        self.task = task

        logger.info("Generating pipelines...")
        dataset, datastore_block = self.datastore.generate_code(dataset, task)
        dataset.check_dataframes(task.target_columns)
        dataset, preprocess_block = self.preprocess.generate_code(dataset, task)
        code_block = datastore_block + preprocess_block
        dataset, sapientml_results = self.generate_code(dataset, task)

        result_pipelines: list[Code] = []
        for pipeline in sapientml_results:
            pipeline.validation = code_block.validation + pipeline.validation
            pipeline.test = code_block.test + pipeline.test
            pipeline.train = code_block.train + pipeline.train
            pipeline.predict = code_block.predict + pipeline.predict
            result_pipelines.append(pipeline)

        logger.info("Executing generated pipelines...")
        executor = PipelineExecutor()
        self.execution_results = executor.execute(
            result_pipelines,
            self.config.initial_timeout,
            Path(dataset.output_dir),
            self.config.cancel,
        )

        logger.info("Evaluating execution results of generated pipelines...")
        lower_is_better = self.task.adaptation_metric in metric_lower_is_better
        self.evaluate(self.execution_results, lower_is_better)

        return (self._best_pipeline, self._best_pipeline_score), self._candidate_scripts

    def generate_code(self, dataset: Dataset, task: Task) -> Tuple[Dataset, list[SimplePipeline]]:
        """
        Generate code including preprocessing and machine learning models using meta-features.

        Parameters
        ----------
        dataset : Dataset
            Dataset class.
        task : Task
            Task class.

        Returns
        ----------
        dataset : Dataset
            Dataset class.
        pipelines : list[SimplePipeline]
            Generated code. Different pipelines contain different models.
        """
        df = dataset.training_dataframe
        # Generate the meta-features
        logger.info("Generating meta features...")
        dataset_summary = summarize_dataset(df, task)  # type: ignore

        labels = predict(task, dataset_summary)
        adapt = Adaptation(
            labels,
            task,
            dataset_summary,
            self.config,
        )
        pipelines = adapt.run_adaptation()

        return dataset, pipelines

    def evaluate(self, execution_results: list[tuple[Code, RunningResult]], lower_is_better: bool = False) -> None:
        """Evaluate execution results of generated pipelines.

        Parameters
        ----------
        execution_results : list[tuple[Code, RunningResult]]
            Execution results containing executed code and resulted output texts.
        lower_is_better : bool, default False
            Specify reverse=(not lower_is_better) for the argument of the sorted method.

        Returns
        ----------
        None

        """
        self._best_pipeline = None
        self._best_pipeline_score = PipelineResult(score=None, metric=None, best_params=None)
        candidate_scripts = []
        for pipeline, result in execution_results:
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

    def save(self, output_dir: Union[Path, str]):
        """Save the script to output_dir.

        Parameters
        ----------
        output_dir : Union[Path, str]
            Output directory path.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        candidate_scripts = self._candidate_scripts
        if candidate_scripts:
            if self._best_pipeline:
                script_body = self._best_pipeline.test
                with open(
                    self.output_dir / add_prefix("final_script.py", self.config.project_name), "w", encoding="utf-8"
                ) as f:
                    f.write(script_body)

                script_body = self._best_pipeline.train
                with open(
                    self.output_dir / add_prefix("final_train.py", self.config.project_name), "w", encoding="utf-8"
                ) as f:
                    f.write(script_body)

                script_body = self._best_pipeline.predict
                with open(
                    self.output_dir / add_prefix("final_predict.py", self.config.project_name), "w", encoding="utf-8"
                ) as f:
                    f.write(script_body)

                with open(
                    self.output_dir / (add_prefix("final_script", self.config.project_name) + ".out.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(self._best_pipeline_score.__dict__, f, cls=JSONEncoder, indent=4)
            else:
                logger.warning("All candidate scripts failed. Final script is not saved.")
                raise RuntimeError("All candidate scripts failed. Final script is not saved.")

            # copy libs
            lib_path = self.output_dir / "lib"
            lib_path.mkdir(exist_ok=True)

            eps = entry_points(group="sapientml.export_modules")
            for ep in eps:
                if ep.name in [self.__class__.__name__, "sample-dataset"]:
                    for file in glob(f"{ep.load().__path__[0]}/*.py"):
                        copyfile(file, lib_path / Path(file).name)

            for index, (script, detail) in enumerate(candidate_scripts, start=1):
                script_body = script.validation
                with open(self.output_dir / f"{index}_script.py", "w", encoding="utf-8") as f:
                    f.write(script_body)

        self.debug_info = {}
        for i, candidate in enumerate(candidate_scripts, start=1):
            info = {"content": candidate[0].model_dump(), "run_info": candidate[1].__dict__}
            self.debug_info[i] = info

        if self.config.debug:
            with open(
                self.output_dir / add_prefix("run_info.json", self.config.project_name), "w", encoding="utf-8"
            ) as f:
                json.dump(self.debug_info, f, cls=JSONEncoder, indent=4)

        if self.config.add_explanation:
            self.add_explanation()

    def add_explanation(self):
        """Call the explain function."""
        explain(
            visualization=True,
            eda=True,
            dataframe=self.dataset.training_dataframe,
            script_path=(self.output_dir / add_prefix("final_script.py", self.config.project_name))
            .absolute()
            .as_posix(),
            target_columns=self.task.target_columns,
            problem_type=self.task.task_type,
            ignore_columns=self.dataset.ignore_columns,
            skeleton=self._best_pipeline.labels,
            explanation=self._best_pipeline.pipeline_json,
            run_info=self.debug_info,
            internal_execution=True,
            timeout=self.config.timeout_for_test,
            cancel=self.config.cancel,
        )
