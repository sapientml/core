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

import pandas as pd
from sapientml.util.logging import setup_logger
from sapientml_core import internal_path
from sapientml_core.meta_features import (
    collect_labels,
    compute_model_meta_features,
    compute_pp_meta_features,
    real_feature_preprocess,
)
from sapientml_core.training.project_corpus import ProjectCorpus
from sapientml_core.util import file_util
from tqdm import tqdm

logger = setup_logger()


# Get the stored features summary
def get_feature_usage_summary():
    """Open and load feature_analysis_summary.json file.

    Returns
    -------
    current_summary : dict
        Loaded summary file
    """
    try:
        with open(internal_path.training_cache / "feature_analysis_summary.json", "r", encoding="utf-8") as f:
            current_summary = json.load(f)
    except Exception:
        current_summary = {}
    return current_summary


def get_column_usage_summary_in_pipeline(pipeline_path, feature_usage_summary):
    """Extract used_cols in pipeline.

    Parameters
    ----------
    pipeline_path :
        pipeline path
    feature_usage_summary : dict
        Summary data

    Returns
    -------
    None
        If pipeline_path not in feature_usage_summary.
    used_columns : pd.Series
        Extractetd columns
    """
    if pipeline_path in feature_usage_summary:
        pipeline_summary = feature_usage_summary[pipeline_path]
    else:
        return None

    if "status" in pipeline_summary and pipeline_summary["status"] == "FINALIZED":
        used_columns = pipeline_summary["used_cols"]
    else:
        used_columns = None

    return used_columns


def get_dataset_folder_name(file_path):
    notebook_info_path = file_path.replace(".py", ".info.json")
    dataset_folder_name = ""
    try:
        with open(notebook_info_path, "r", encoding="utf-8") as notebook_info_file:
            notebook_info = json.load(notebook_info_file)
            if isinstance(notebook_info, list):
                notebook_info = notebook_info[1]

            if isinstance(notebook_info, dict):
                dataset_folder_name = notebook_info["dataset_folder"]
    except Exception:
        logger.warning("Could not read JSON info file: {}".format(notebook_info_path))
    return dataset_folder_name


def collect_training_meta_feature(mode):
    """Read csv and Generate the meta-features.

    Parameters
    ----------
    mode : str
        "clean" or "as-is"

    Returns
    -------
    final_pp_meta_features : pd.Dataframe
        Dataframe of pp_meta_features and labels
    final_model_meta_feature_df : pd.Dataframe
        Dataframe of model_meta_feature and models
    """
    pp_meta_features = []
    model_meta_features = []
    feature_usage_summary = get_feature_usage_summary()

    corpus = ProjectCorpus()
    projects = corpus.project_list

    total_number_target_pipelines = len(projects)

    column_not_found = 0

    for i in tqdm(range(0, total_number_target_pipelines)):
        project = projects[i]
        pipeline_path = project.pipeline_path
        file_name = projects[i].file_name

        dataset_folder_name = get_dataset_folder_name(pipeline_path)
        project.csv_name = dataset_folder_name + "_" + project.csv_name

        dataset_path = project.dataset_path
        logger.debug(
            "EXTRACTING META-FEATURES:", i + 1, "out of ", total_number_target_pipelines, "PIPELINE:", pipeline_path
        )
        try:
            df = file_util.read_csv(
                Path(project.dataset_path),
                Path(project.pipeline_path),
            )
            if mode != "as-is":
                import copy

                df_copy = copy.deepcopy(df)
                used_cols = get_column_usage_summary_in_pipeline(file_name, feature_usage_summary)
                if used_cols is not None:
                    df = df[used_cols]
        except KeyError:
            column_not_found += 1
            df = df_copy
        except Exception:
            logger.warning("Could not read CSV: {}".format(dataset_path))
            continue

        df.rename(columns=lambda x: x.strip(), inplace=True)

        proj_name = pipeline_path
        target_column_name = projects[i].target_column_name
        # Generate the meta-features
        pp_meta_feature_dict = compute_pp_meta_features(df, proj_name, project, target_column_name)
        model_meta_feature_dict = compute_model_meta_features(df, proj_name, project, target_column_name)

        if pp_meta_feature_dict is not None:
            pp_meta_features.append(pp_meta_feature_dict)

        if model_meta_feature_dict is not None:
            model_meta_features.append(model_meta_feature_dict)

    pp_meta_features_df = pd.DataFrame(pp_meta_features)
    model_meta_feature_df = real_feature_preprocess(pd.DataFrame(model_meta_features))

    # Add the labels
    annotated_notebooks_path = internal_path.project_labels_path
    labels_df = collect_labels(annotated_notebooks_path)

    # Merge two dataframes
    final_pp_meta_features = pd.merge(pp_meta_features_df, labels_df, on="file_name", how="inner")
    logger.debug(f"final_pp_meta_features shape:{final_pp_meta_features.shape}")

    models = []
    is_clf = []

    for _, row in labels_df.iterrows():
        model_added = False
        for col in labels_df.columns.to_list():
            if isinstance(col, str) and col.startswith("MODEL:") and row[col] == 1:
                models.append(col)
                # 0: Regresssor 1: Classifier
                clf = 0 if ":Regressor:" in col else 1
                is_clf.append(clf)
                model_added = True
                break
        if not model_added:
            models.append("MODEL:NOT_DETECTED")
            is_clf.append(2)

    labels_df["target"] = models

    model_df = labels_df[["file_name", "target"]]
    model_df["feature:is_clf"] = is_clf
    model_df["weight"] = 1
    # Merge two dataframes
    final_model_meta_feature_df = pd.merge(model_meta_feature_df, model_df, on="file_name", how="inner")
    logger.debug(f"final_model_meta_feature_df shape:{final_model_meta_feature_df.shape}")

    return final_pp_meta_features, final_model_meta_feature_df


def main():
    """
    This main function calls the subfunctions and outputs the result to csv.

    Description of mode : "clean" | "as-is"
        as-is mode: compute meta-features based on all the meta-features in the dataset
        clean mode: only use the meta-features that are used in the pipeline

    """
    mode = "clean"
    final_pp_meta_features_df, final_model_meta_feature_df = collect_training_meta_feature(mode)
    pp_meta_features_path = internal_path.training_cache / "pp_metafeatures_training.csv"
    model_meta_features_path = internal_path.training_cache / "model_metafeatures_training.csv"
    final_pp_meta_features_df.to_csv(pp_meta_features_path, index=False)
    final_model_meta_feature_df.to_csv(model_meta_features_path, index=False)
    logger.info(f"Meta-data successsfully stored at:{internal_path.training_cache}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="Tag for output files and dirs.")
    args = parser.parse_args()
    if args.tag:
        internal_path.training_cache = internal_path.training_cache / args.tag
    main()
