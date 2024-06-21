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


import json
from pathlib import Path

from sapientml.util.logging import setup_logger
from sapientml_core import internal_path
from sapientml_core.training import project_corpus
from sapientml_core.training.denoising import ast_info_collector as collector
from sapientml_core.util import file_util

logger = setup_logger()


def extract(json_metadata_file):
    """Extracting the pipeline.

       This function is collecting the pipeline details and extract
       the target column based on file data structure.

    Parameters
    ----------
    json_metadata_file : str
        The parameter containg each pipeline details.

    Returns
    -------
    str
       This funtion will return target_column_name.

    Raises
    ------
        The ``Raises`` section is a list of all exceptions
        that are relevant to the interface.

    """
    with open(json_metadata_file, "r", encoding="utf-8") as f:
        notebook_info = json.load(f)

    if isinstance(notebook_info, dict):
        target_column_name = notebook_info["target_column_name"]
    elif isinstance(notebook_info, list):
        target_column_name = notebook_info[1]["target_column_name"]
    else:
        logger.warning("Wrong format: {}".format(json_metadata_file))
        raise

    return target_column_name


def main(test_mode=False):
    """Fetch all the pipeline details from corpus and parse it using libcst library.

       This script performs static analysis of the pipeline to identify
       if there is any explicit renaming of the column names or explicit
       deletion of columns in the pipeline and create static_info.json file.

    Parameters
    ----------
    test_mode : bool
         This parameter is used for test mode.

    Raises
    ------
    Exception:
        The ``Raises`` section is a list of all exceptions
        that are relevant to the interface.

    """
    corpus = project_corpus.ProjectCorpus()
    projects = corpus.project_list
    static_info_map = {}

    total_number_target_pipelines = len(projects)

    for i in range(0, total_number_target_pipelines):
        if test_mode and i > 5:
            break
        logger.info(f"RUNNING:{i + 1} out of:{total_number_target_pipelines} PIPELINE:{projects[i].pipeline_path}")
        project = projects[i]
        pipeline = project.pipeline_path
        file_name = project.file_name

        static_info = {}
        try:
            dataset = file_util.read_csv(
                Path(project.dataset_path),
                Path(project.pipeline_path),
            )
        except Exception:
            raise

        json_meta = pipeline.replace(".py", ".info.json")

        target = extract(json_meta)
        source_file = pipeline
        with open(source_file, "r", encoding="utf-8") as f:
            source = f.read()

        try:
            column_api_map = collector.get_column_api_map(source)
        except Exception:
            raise

        dataset_columns = list(dataset.columns)
        dropped_columns = []
        renamed_columns = []
        for column in column_api_map:
            if "drop" in column_api_map[column]:
                if column != target and column in dataset_columns:
                    dropped_columns.append(column)
            if "rename" in column_api_map[column]:
                renamed_columns.append(column)

        static_info["drop_api"] = dropped_columns
        static_info["rename_api"] = renamed_columns
        static_info["target"] = target
        static_info_map[file_name] = static_info
        try:
            dataset.drop(dropped_columns, axis=1, inplace=True)
        except Exception:
            raise

    logger.info(f"Total number of notebooks: {len(static_info_map)}")
    with open(internal_path.training_cache / "static_info.json", "w", encoding="utf-8") as f:
        json.dump(static_info_map, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="Tag for output files and dirs.")
    args = parser.parse_args()
    if args.tag:
        internal_path.training_cache = internal_path.training_cache / args.tag
    test_mode = False
    main(test_mode)
