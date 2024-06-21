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


import doctest
import json
import re
from pathlib import Path

from sapientml.util.logging import setup_logger
from sapientml_core import internal_path
from tqdm import tqdm

from .project import ProjectInfo

logger = setup_logger()


class ProjectCorpus:
    def __init__(self, target_project_name_list=None):
        self.target_project_name_list = target_project_name_list
        self.clean_notebook_dir_path = internal_path.clean_dir
        self.dataset_dir_path = internal_path.corpus_path / "dataset"
        self.metadata_dir_path = internal_path.corpus_path / "metadata"
        self.project_list = self._extract_project_info()

    def _extract_project_info(self):
        project_list = []

        if self.target_project_name_list:
            pipeline_file_names = [Path(project_path) for project_path in self.target_project_name_list]
        else:
            pipeline_file_names = Path(self.clean_notebook_dir_path).rglob("*.py")

        for notebook_path in tqdm(list(pipeline_file_names)):
            notebook_info_path = notebook_path.with_suffix(".info.json")
            notebook_name = notebook_path.stem
            logger.debug(f"Extracting Project Info for {notebook_name}")
            # Read the target column information
            try:
                with open(notebook_info_path, "r", encoding="utf-8") as notebook_info_file:
                    notebook_info = json.load(notebook_info_file)
            except Exception:
                logger.warning("Could not read JSON info file: {}".format(notebook_info_path))
                continue

            if isinstance(notebook_info, list):
                notebook_info = notebook_info[1]

            if isinstance(notebook_info, dict):
                target_column_name = notebook_info["target_column_name"]
                dataset_folder_name = notebook_info["dataset_folder"]
                accuracy = notebook_info["accuracy"]
                metric = "accuracy"
                if accuracy == "N/A":
                    accuracy = notebook_info["r2"]
                    metric = "r2"
                try:
                    accuracy = float(accuracy[:-1])  # discarding the percentage (%) sign from the end
                except Exception:
                    accuracy = 0
            else:
                logger.warning("Wrong format: {}".format(notebook_info_path))
                continue

            if isinstance(target_column_name, str):
                if target_column_name == "UNKNOWN":
                    continue
            elif isinstance(notebook_info, list):
                if target_column_name[0] == "UNKNOWN":
                    continue
            # Read the dataset
            project_fqn = notebook_name + ".py"
            dataset_paths = [
                p
                for p in (Path(self.dataset_dir_path) / dataset_folder_name).glob("*")
                if re.search(r"/*\.(csv|tsv)", str(p))
            ]
            if len(dataset_paths) == 0:
                logger.warning(
                    "Could not find CSV/TSV file under {}/{}".format(self.dataset_dir_path, dataset_folder_name)
                )
                continue

            dataset_path = dataset_paths[0]
            dataset_name = dataset_path.stem
            if len(dataset_paths) > 1:
                logger.warning(
                    "Found multiple CSV/TSV files under {}. Using {}...".format(
                        self.clean_notebook_dir_path, dataset_name
                    )
                )

            project_info = ProjectInfo(
                str(notebook_path),
                str(dataset_path),
                project_fqn,
                notebook_name,
                accuracy,
                dataset_name,
                target_column_name,
                metric,
            )
            project_list.append(project_info)
        return project_list


if __name__ == "__main__":
    doctest.testmod()
