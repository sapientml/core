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


import pickle
from collections import OrderedDict, defaultdict
from typing import Literal

import pandas as pd
from sapientml.util.logging import setup_logger
from sapientml_core import internal_path
from sapientml_core.design import search_space
from sapientml_core.training import meta_feature_selector
from sklearn.tree import DecisionTreeClassifier

logger = setup_logger()


def train_p_model(X, y):
    """Build a decision tree classifier from the training set (X, y).

    Parameters
    ----------
    X : MatrixLike = np.ndarray | pd.DataFrame | spmatrix   |
        ArrayLike = numpy.typing.ArrayLike
        The training input samples.
    y : MatrixLike = np.ndarray | pd.DataFrame | spmatrix   |
        ArrayLike = numpy.typing.ArrayLike
        The target values (class labels) as integers or strings

    Returns
    -------
    model : DecisionTreeClassifier
        Fitted estimator.
    """
    model = DecisionTreeClassifier(class_weight="balanced", max_depth=3)
    model.fit(X, y)
    return model


def _train_preprocessors(train_data, feature_selection: Literal["select_manually", "customized"]):
    logger.info("Training skeleton predictor for preprocessors...")
    data = train_data
    data.drop(
        data.filter(regex="(TEMPLATE|IGNORE|EVAL:|RPEPROCESS:|MODEL:|Unnamed:)").columns,
        axis=1,
        inplace=True,
    )
    data["project_target"] = (
        data["csv_name"] + "_" + data["target_column_name"].apply(lambda line: "_".join(sorted(eval(line))))
    )
    all_labels = [v for v in data.columns if v.startswith(("PREPROCESS:"))]
    second_to_full_labels = defaultdict(list)
    for label in all_labels:
        second_to_full_labels["PREPROCESS:" + label.split(":")[1]].append(label)

    pp_models = OrderedDict()

    selected_features_map = meta_feature_selector.select_based_on_correlation(data)

    for _, detail_labels in second_to_full_labels.items():
        for label in detail_labels:
            logger.debug(label)
            main_df = data.copy()
            # Feature Selection On
            y = main_df[label]
            X = main_df[search_space.meta_feature_list]

            if feature_selection == "select_manually":
                selected_features = meta_feature_selector.select_features(label)
                logger.debug("Selected Features:", selected_features)
                X = main_df[selected_features]
            elif feature_selection == "customized":
                selected_features = selected_features_map[label]
                if len(selected_features) == 0:
                    selected_features = meta_feature_selector.select_sequentially(X, y)
                logger.debug("Selected Features:", selected_features)
                X = main_df[selected_features]

            pp_model = train_p_model(X, y)
            pp_models[label] = (pp_model, selected_features)

    return pp_models


def _prepare_model_training_data(raw_meta_feature_train):
    # Remove all the unnecessary meta-features
    final_meta_features = raw_meta_feature_train[search_space.project_related_metadata + search_space.meta_feature_list]
    final_meta_features.fillna(0, inplace=True)
    for semantic_label, columns in search_space.label_mapping.items():
        try:
            final_meta_features[semantic_label] = raw_meta_feature_train[columns].sum(axis=1)
            final_meta_features[semantic_label] = final_meta_features[semantic_label].apply(lambda x: 1 if x > 0 else 0)
        except KeyError as e:
            logger.warning(e)

    return final_meta_features


def main():
    """This main function preprocesses the learning data and saves fitted estimator for the DecisionTreeClassifier.

    Description of feature_selection : "select_manually" | "customized"
        Specify how features are selected.
    """
    training_data_path = internal_path.training_cache / "pp_metafeatures_training.csv"
    # "select_manually" | "customized"
    feature_selection = "customized"
    raw_meta_feature_train = pd.read_csv(training_data_path)
    meta_feature_train = _prepare_model_training_data(raw_meta_feature_train)
    pp_models = _train_preprocessors(meta_feature_train, feature_selection)
    # Save model
    with open(internal_path.training_cache / "pp_models.pkl", "wb") as f:
        pickle.dump(pp_models, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="Tag for output files and dirs.")
    args = parser.parse_args()
    if args.tag:
        internal_path.training_cache = internal_path.training_cache / args.tag

    main()
