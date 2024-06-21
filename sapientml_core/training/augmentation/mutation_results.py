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

import os
from collections import OrderedDict, defaultdict

import pandas as pd
from sapientml_core import internal_path
from sapientml_core.seeding.predictor import name_to_label_mapping
from sapientml_core.training.project_corpus import ProjectCorpus
from tqdm import tqdm


class MutationResult:
    """MutationResult class.

    This class loads the mutated results for each pipeline that were already stored in the sapientml_core cache
    and combines all the results in a CSV file and selects the best model.

    """

    def __init__(self, mutation_result_path, project_list):
        self.mutation_result_path = mutation_result_path
        self.project_list = project_list

    def load_results(self):
        """Collects the score for augmented pipelines from exec_info directory.

        Returns
        -------
        results: defaultdict

        """
        results = defaultdict(defaultdict)
        models = list(name_to_label_mapping.keys()) + ["original"]
        execution_root_dir = internal_path.training_cache / "exec_info"

        for i in tqdm(range(0, len(self.project_list))):
            project = self.project_list[i]
            project_exec_dir = execution_root_dir / project.notebook_name
            project_key = project.file_name
            for model in models:
                result_file_path = project_exec_dir / model / "stdout.txt"
                acc, r2 = 0, 0
                if not os.path.exists(result_file_path):
                    results[project_key][model] = 0
                    continue
                with open(result_file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        for trail in ["Accuracy: ", "R2: "]:
                            data = line
                            if data.count(trail) > 0:
                                data = data[data.index(trail) + len(trail) :].strip()
                                if data.count("%") > 0:
                                    data = data[: data.index("%")]
                                    data = float(data) / 100
                                    if trail == "Accuracy: ":
                                        acc = data
                                    if trail == "R2: ":
                                        r2 = data
                    if project.metric == "accuracy":
                        results[project_key][model] = round(acc, 5)
                    elif project.metric == "r2":
                        results[project_key][model] = round(r2, 5)

            best_models = []
            sorted_results = sorted(results[project_key].items(), key=lambda x: x[1], reverse=True)
            best_value = 0
            for model, value in sorted_results:
                if value > 0 and value >= best_value:
                    best_models.append(model)
                    best_value = value
                else:
                    break

            results[project_key]["best_models"] = best_models

        return results


def main():
    """Fetch the augmented pipeline results and store it in mutation_results.csv."""
    corpus = ProjectCorpus()  # Fetch all project and pipeline details
    mutation_result = MutationResult(internal_path.training_cache, corpus.project_list)
    results = mutation_result.load_results()
    result_list = []
    for key, result in results.items():
        result["file_name"] = key
        result = OrderedDict(result)
        result.move_to_end("file_name", last=False)
        result_list.append(result)
    result_dataframe = pd.DataFrame(result_list)
    result_dataframe.to_csv(internal_path.training_cache / "mutation_results.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="Tag for output files and dirs.")
    args = parser.parse_args()
    if args.tag:
        internal_path.training_cache = internal_path.training_cache / args.tag

    main()
