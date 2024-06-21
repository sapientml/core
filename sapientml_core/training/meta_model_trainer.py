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
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from sapientml.util.logging import setup_logger
from sapientml_core import internal_path
from sapientml_core.design.label_util import map_label_to_name, name_to_label_mapping
from sapientml_core.training.augmentation.mutation_results import MutationResult
from sapientml_core.training.project_corpus import ProjectCorpus
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

logger = setup_logger()

pd.set_option(
    "display.max_rows",
    None,
    "display.max_columns",
    None,
)


def evaluate_on_training_data(probas, Y, weights, class_names, top_K=3):
    """evaluate_on_training_data.

    Parameters
    ----------
    probas : np.ndarray
    Y : pandas.Series
    weights : pandas.Series
    class_names : np.ndarray
    top_K : int, default 3

    """
    Y = np.array(Y)

    top_1_pred = probas.argmax(axis=1)

    tot_w = np.sum(weights)

    top_1_tot = np.sum([x * y for x, y in zip(class_names[top_1_pred] == Y, weights)])

    top_K_pred = probas.argsort(axis=1)[:, ::-1][:, :top_K]
    top_K_tot = np.sum([x * y for x, y in zip((class_names[top_K_pred] == Y.reshape(-1, 1)).sum(axis=1), weights)])

    logger.info(f"Top-1 Acc: {top_1_tot / tot_w * 100.:.3f} %")
    logger.info(f"Top-{top_K} Acc: {top_K_tot / tot_w * 100.:.3f} %")


def _predict_models(m_model, task_type, test_meta_features_batch):
    test_meta_features_batch = test_meta_features_batch.fillna(0)

    lr_proba = m_model[0].predict_proba(test_meta_features_batch)
    svc_proba = m_model[1].predict_proba(test_meta_features_batch)

    predict_probabilities_batch = lr_proba + svc_proba

    probability_df = pd.DataFrame(predict_probabilities_batch, columns=m_model[0].classes_)

    with open(internal_path.training_cache / "predicted_models_proba.csv", "w", encoding="utf-8") as f:
        probability_df.to_csv(f, index=False)
    ranked_model_indices_batch = predict_probabilities_batch.argsort(axis=1)[:, ::-1]
    ranked_model_normalized_labels_batch = np.array(m_model[0].classes_)[ranked_model_indices_batch]

    top_3_prediction_actual_label_batch = []
    all_appeared_models = set()
    for preds, task_type in zip(ranked_model_normalized_labels_batch, task_type):
        top_3_prediction_one_subject = []
        for item in preds:
            if item in name_to_label_mapping and task_type in name_to_label_mapping[item]:
                actual_model_label = name_to_label_mapping[item][task_type]
                all_appeared_models.add(actual_model_label)
                top_3_prediction_one_subject.append(actual_model_label)
                if len(top_3_prediction_one_subject) == 3:
                    break

        top_3_prediction_actual_label_batch.append(top_3_prediction_one_subject)
    logger.info(all_appeared_models)
    logger.info(f"Unique Models:{len(all_appeared_models)}")
    return top_3_prediction_actual_label_batch


def print_out_of_scope_models(dataframe, label_to_name):
    i = 0
    unused_models = []
    for _, row in dataframe.iterrows():
        if row["target"] not in label_to_name:
            unused_models.append(row["target"])
            logger.debug(i, row["file_name"], row["target"])
            i += 1
    logger.debug(dict(Counter(unused_models)))


def _prepare_model_training_data_augmented(
    dataframe, take_only_best_notebook_for_dataset=False, accuracy_threshold=0.5
):
    augmented_meta_feature_path = internal_path.training_cache / "model_metafeatures_augmented.csv"

    if os.path.exists(augmented_meta_feature_path):
        augmented_df = pd.read_csv(augmented_meta_feature_path)
    else:
        corpus = ProjectCorpus()
        mutation_result = MutationResult(internal_path.training_cache, corpus.project_list)
        results = mutation_result.load_results()

        label_to_name = map_label_to_name()

        dataframe["normalized_target"] = [
            label_to_name[label] if label in label_to_name else "MODEL:OUT_OF_SCOPE" for label in dataframe["target"]
        ]

        origninal_model_map = (
            dataframe[["file_name", "normalized_target"]].set_index("file_name").to_dict()["normalized_target"]
        )

        records = dataframe.to_dict("records")

        augmented_records = []
        weights = []

        for record in records:
            project_key = record["file_name"]
            result = results[project_key]
            best_models = result["best_models"]

            if "original" in best_models:
                normalized_target = origninal_model_map[project_key]
                if normalized_target in best_models:
                    best_models.remove("original")

            if best_models:
                for best_model in best_models:
                    new_record = record.copy()
                    normalized_target = best_model
                    accuracy = result[best_model]
                    new_record["normalized_target"] = normalized_target
                    new_record["accuracy"] = accuracy
                    weights.append(1 / len(best_models))
                    augmented_records.append(new_record)
            else:
                weights.append(1)
                augmented_records.append(record.copy())

        augmented_df = pd.DataFrame(augmented_records)
        augmented_df["weight"] = weights

        augmented_df.drop(augmented_df[augmented_df["normalized_target"] == "MODEL:OUT_OF_SCOPE"].index, inplace=True)
        augmented_df.to_csv(internal_path.training_cache / "model_metafeatures_augmented.csv", index=False)

    if take_only_best_notebook_for_dataset:
        idx = augmented_df.groupby(["csv_name"])["accuracy"].transform(max) == augmented_df["accuracy"]
        augmented_df = augmented_df[idx]

    if accuracy_threshold is not None:
        augmented_df = augmented_df[(augmented_df["accuracy"] >= accuracy_threshold)]
    augmented_df.to_csv(internal_path.training_cache / "model_metafeatures_augmented_x.csv", index=False)
    X = augmented_df[[x for x in dataframe.columns if x.startswith("feature:")]]
    X = X.fillna(0)
    weights = augmented_df["accuracy"]
    y = augmented_df["normalized_target"]
    logger.info(f"Number of Training Samples:{X.shape}")
    logger.info(Counter(y))
    return X, y, weights


def test_prediction(m_models, test_df_path):
    """Predict the model with test data.

    Parameters
    ----------
    m_models : tuple
    test_df_path : PosixPath

    """
    test_df = pd.read_csv(test_df_path)

    meta_features = test_df[[x for x in test_df.columns if x.startswith("feature:")]]
    task_type = ["c" if x == 1 else "r" for x in test_df["feature:is_clf"].to_list()]

    predict_results = _predict_models(m_models, task_type, meta_features)
    prediction_df = pd.DataFrame(predict_results)
    prediction_df.insert(0, "project_name", test_df["project_name"].to_list())
    with open(internal_path.training_cache / "predicted_models_debug.csv", "w", encoding="utf-8") as f:
        prediction_df.to_csv(f, index=False)


def training(training_data_path, model_dir_path_default):
    """Train the model and save them as pickle file.

    Parameters
    ----------
    training_data_path : PosixPath
    model_dir_path_default : PosixPath

    Returns
    -------
    clf_1 : LogisticRegression
    clf_2 : SVC

    """
    meta_features = pd.read_csv(training_data_path)
    take_only_best_notebook_for_dataset = True
    accuracy_threshold = 0.5
    X, y, weights = _prepare_model_training_data_augmented(
        meta_features, take_only_best_notebook_for_dataset, accuracy_threshold
    )

    clf_1, clf_2 = train_meta_models(X, weights, y)
    model_feature_weights = {}
    for i, coef in enumerate(clf_1.coef_):
        logger.debug(clf_1.classes_[i])
        normalized_co_ef = np.std(X, 0) * coef
        mf_weight_pairs = []
        for j, n_co_ef in enumerate(normalized_co_ef):
            mf_weight_pairs.append((X.columns[j], n_co_ef))

        mf_weight_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        model_feature_weights[clf_1.classes_[i]] = mf_weight_pairs

    evaluate_on_training_data(clf_1.predict_proba(X) + clf_2.predict_proba(X), y, weights, clf_1.classes_)

    with open(model_dir_path_default / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(model_feature_weights, f)

    with open(model_dir_path_default / "mp_model_1.pkl", "wb") as f1:
        with open(model_dir_path_default / "mp_model_2.pkl", "wb") as f2:
            pickle.dump(clf_1, f1)
            pickle.dump(clf_2, f2)
    logger.info(f"model weights saved to:{internal_path.training_cache}")

    return clf_1, clf_2


def train_meta_models(X, weights, y):
    """Train the model using LogisticRegression and SVC.

    Parameters
    ----------
    X : DataFrame
    weights : pandas.Series
    y : pandas.Series

    Returns
    -------
    clf_1 : LogisticRegression
    clf_2 : SVC

    """
    clf_1 = LogisticRegression(max_iter=100, random_state=42)
    clf_2 = SVC(kernel="rbf", probability=True, random_state=42)
    clf_1.fit(X, y, sample_weight=weights)
    clf_2.fit(X, y, sample_weight=weights)
    assert all([clf_1.classes_[i] == clf_2.classes_[i] for i in range(len(clf_1.classes_))])
    return clf_1, clf_2


def main():
    """This is a main function.

    This script trains the meta-models, i.e., skeleton predictor -- for both
    preprocess components and model components.

    """
    training_data_path = internal_path.training_cache / "model_metafeatures_training.csv"
    test_data_path = internal_path.model_path / "model_metafeatures_test.csv"

    models = training(training_data_path, internal_path.training_cache)
    test_prediction(models, test_data_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="Tag for output files and dirs.")
    args = parser.parse_args()
    if args.tag:
        internal_path.training_cache = internal_path.training_cache / args.tag

    main()
