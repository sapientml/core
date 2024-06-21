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
import os

from sapientml.util.logging import setup_logger
from sapientml_core import internal_path
from sapientml_core.training import project_corpus
from sapientml_core.util import file_util

logger = setup_logger()


def compare_basic(project, static_info):
    """This function create summary for each pipeline from dataset_snapshot.

    Parameters
    ----------
    project : ProjectInfo
        This contain information of all the pipelines.

    static_info : dict
        The parameter contains {'drop_api': ['Name'], 'rename_api': [], 'target': 'Legendary'}

    Returns
    -------
    dict
       {
            'used_cols': ['Generation', 'Legendary'],
            'unmapped_cols': ['Type 2', 'Attack'],
            'new_cols': ['Fire', 'Water', 'Bug'],
            'target': 'Legendary',
            'deleted': ['Name'],
            'status': 'IN_PROGRESS',

        }

    """
    # load Initial Dataframe and Final Dataframe
    summary = {}
    project_root = internal_path.clean_dir
    pipeline_relative_path = project.pipeline_path[len(str(project_root)) + 1 :]
    feature_processing_history_file_path = (
        internal_path.training_cache / "dataset-snapshots" / pipeline_relative_path.replace(".py", "_col_dumper.json")
    )

    if not os.path.exists(feature_processing_history_file_path):
        return summary

    feature_processing_history = file_util.load_json(feature_processing_history_file_path)

    column_snapshots = []
    for _, column_info in feature_processing_history[0].items():
        if column_info[2] == "<class 'pandas.core.frame.DataFrame'>":
            if len(column_info[0]) > 1:
                no_int_present = True
                for column_name in column_info[0]:
                    if isinstance(column_name, int):
                        no_int_present = False
                        break
                if no_int_present:
                    column_snapshots.append(column_info[0])

    if len(column_snapshots) == 0:
        return summary

    # compare shapes
    summary["pipeline"] = project.file_name

    initial_features = set(column_snapshots[0])

    final_features = set(column_snapshots[-1])
    target = static_info.get("target", "n/a")
    if isinstance(target, list):
        final_features |= set(target)
    else:
        final_features.add(target)

    used_columns = final_features.intersection(initial_features)
    unmapped_cols = initial_features.difference(used_columns)
    new_cols = final_features.difference(used_columns)

    # column expansion detection
    column_expansion = set()
    remove_from_new_cols = set()
    for unmapped in unmapped_cols:
        for new_col in new_cols:
            if new_col.startswith(unmapped):
                column_expansion.add(unmapped)
                remove_from_new_cols.add(new_col)
                used_columns.add(unmapped)

    unmapped_cols = unmapped_cols.difference(column_expansion)
    new_cols = new_cols.difference(remove_from_new_cols)

    # renaming handling
    possibly_renamed = set()
    renamed_cols = static_info["rename_api"]
    for unmapped_col in unmapped_cols:
        if unmapped_col in renamed_cols:
            index = renamed_cols.index(unmapped_col)
            renamed = renamed_cols[index + 1]
            if renamed in new_cols:
                possibly_renamed.add(unmapped_col)
                new_cols.remove(renamed)
                used_columns.add(unmapped_col)

    unmapped_cols = unmapped_cols.difference(possibly_renamed)

    # target handling

    if "0" in new_cols and static_info["target"] in unmapped_cols:
        used_columns.add(target)
        unmapped_cols.remove(target)
        new_cols.remove("0")

    # possible deletion
    possibly_deleted = set()
    deleted_cols = static_info["drop_api"]
    for unmapped_col in unmapped_cols:
        if unmapped_col in deleted_cols:
            possibly_deleted.add(unmapped_col)

    unmapped_cols = unmapped_cols.difference(possibly_deleted)

    # delete index
    if "index" in new_cols:
        new_cols.remove("index")

    summary["used_cols"] = list(used_columns)
    summary["unmapped_cols"] = list(unmapped_cols)
    summary["new_cols"] = list(new_cols)
    summary["target"] = target
    summary["deleted"] = list(possibly_deleted)

    if len(new_cols) == 0 or len(unmapped_cols) == 0:
        summary["status"] = "FINALIZED"
    else:
        summary["status"] = "IN_PROGRESS"
        logger.info(file_util.read_file(project.pipeline_path))

    return summary


def main(test_mode=False):
    """This main function fetch all the pipeline details from json files in
    dataset_snapshots folder and summarize it and store in feature_analysis_summary.json
    Summary consist of following information:
    pipeline name
    used_cols
    unmapped_cols
    new_cols
    target
    deleted
    status

    Parameters
    ----------
    test_mode : bool
        The parameter is using for test mode.

    Raises
    ------
    Exception
            The ``Raises`` section is a list of all exceptions
            that are relevant to the interface.

    """
    # get selected pipelines
    corpus = project_corpus.ProjectCorpus()
    projects = corpus.project_list

    total_number_target_pipelines = len(projects)
    finalized = 0
    error = 0

    current_summary = {}

    try:
        with open(internal_path.training_cache / "static_info.json", "r", encoding="utf-8") as f:
            static_info = json.load(f)
    except Exception:
        raise

    feature_removed = 0

    for i in range(0, total_number_target_pipelines):
        if test_mode and i > 5:
            break
        pipeline = projects[i].pipeline_path
        file_name = projects[i].file_name
        if file_name not in static_info:
            continue

        logger.info(f"COMPARING: {i + 1} out of:{total_number_target_pipelines}, PIPELINE:{pipeline}")
        if file_name in current_summary and current_summary[file_name] is not None:
            status = current_summary[file_name].get("status", "n/a")
            if status == "FINALIZED":
                finalized += 1
                continue
        try:
            summary = compare_basic(projects[i], static_info[file_name])
            status = summary.get("status", "n/a")
            if status == "FINALIZED":
                finalized += 1

            deleted = summary.get("deleted", [])
            unmapped = summary.get("unmapped_cols", [])
            if len(deleted) + len(unmapped) > 0:
                feature_removed += 1
        except Exception:
            summary = {"pipeline": file_name, "status": "ERROR"}
            error += 1
            raise

        current_summary[file_name] = summary

    with open(
        internal_path.training_cache / "feature_analysis_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(current_summary, f, indent=4)

    logger.info(f"FINALIZED: {finalized},Total Pipeline:{len(current_summary)},Feature Removed:{feature_removed}")
    logger.error(f"ERROR:, {error}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="Tag for output files and dirs.")
    args = parser.parse_args()
    if args.tag:
        internal_path.training_cache = internal_path.training_cache / args.tag
    test_mode = False
    main(test_mode)
