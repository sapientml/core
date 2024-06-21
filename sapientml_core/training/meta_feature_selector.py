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


from sapientml_core import ps_macros
from sapientml_core.design import search_space
from sklearn.tree import DecisionTreeClassifier


def select_k_best_features(X, y):
    """Select the top k explanatory variables.

    Parameters
    ----------
    X : MatrixLike = np.ndarray | pd.DataFrame | spmatrix
        The training input samples.
    y : ArrayLike = numpy.typing.ArrayLike
        The target values

    Returns
    -------
    list
        Returns a list of the top k selected column names.
    """
    from sklearn.feature_selection import SelectKBest, mutual_info_regression

    # Select top 2 features based on mutual info regression
    selector = SelectKBest(mutual_info_regression, k=3)
    selector.fit(X, y)
    return list(X.columns[selector.get_support()])


def select_by_rfe(X, y):
    """Extract the top N(=n_features_to_select) feature values of importance by RFE(Recursive Feature Elimination).

    Parameters
    ----------
    X : MatrixLike = np.ndarray | pd.DataFrame | spmatrix   |
        ArrayLike = numpy.typing.ArrayLike
        The training input samples.
    y : ArrayLike = numpy.typing.ArrayLike
        The target values.

    Returns
    -------
    list
        Returns a list of selected column names.
    """
    from sklearn.feature_selection import RFE

    # #Selecting the Best important features according to Logistic Regression
    rfe_selector = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=2, step=1)
    rfe_selector.fit(X, y)
    return list(X.columns[rfe_selector.get_support()])


def select_from_model(X, y):
    """Select features based on importance weights.

    Parameters
    ----------
    X : MatrixLike = np.ndarray | pd.DataFrame | spmatrix
        The training input samples.
    y : None | ArrayLike = numpy.typing.ArrayLike
        The target values(integers that correspond to classes in classification, real numbers in regression).

    Returns
    -------
    list
        Returns a list of selected column names.
    """
    from sklearn.feature_selection import SelectFromModel

    # #Selecting the Best important features according to Logistic Regression using SelectFromModel
    sfm_selector = SelectFromModel(estimator=DecisionTreeClassifier())
    sfm_selector.fit(X, y)
    return list(X.columns[sfm_selector.get_support()])


def select_sequentially(X, y):
    """Select feature quantity in order and select feature quantity by greedy method.

    Parameters
    ----------
    X : MatrixLike = np.ndarray | pd.DataFrame | spmatrix
        Training vectors
    y : None | ArrayLike = numpy.typing.ArrayLike
        Target values. This parameter may be ignored for unsupervised learning.

    Returns
    -------
    list
        Returns a list of selected column names.
    """
    from sklearn.feature_selection import SequentialFeatureSelector

    # Selecting the Best important features according to Logistic Regression
    sfs_selector = SequentialFeatureSelector(
        estimator=DecisionTreeClassifier(), n_features_to_select=3, cv=10, direction="backward"
    )
    sfs_selector.fit(X, y)
    return list(X.columns[sfs_selector.get_support()])


def select_based_on_correlation(data):
    """Create correlation maps for learning data.

    Parameters
    ----------
    data : dataframe
        Training data

    Returns
    -------
    correlation_map : defaultdict(list)
    """
    from collections import defaultdict

    corr = data.corr(numeric_only=True)
    correlation_map = defaultdict(list)
    for i in range(len(corr.columns)):
        left = corr.columns[i]
        for j in range(i):
            if corr.iloc[i, j] >= 0.25:
                right = corr.columns[j]
                if left[0] != right[0]:
                    correlation_map[left].append(right)

        if len(correlation_map[left]) == 0:
            for j in range(i):
                if corr.iloc[i, j] >= 0.15:
                    right = corr.columns[j]
                    if left[0] != right[0]:
                        correlation_map[left].append(right)

        if len(correlation_map[left]) == 0:
            correlation_map[left] = list(search_space.meta_feature_list)
    return correlation_map


def select_features(label):
    """Return manually selected feature labels.

    Parameters
    ----------
    label : str

    Returns
    -------
    selection_model[label] : list
    """
    selection_model = {
        ps_macros.FILL: [ps_macros.MISSING_PRESENCE],
        # ps_macros.DROP: [ps_macros.MISSING_PRESENCE],
        ps_macros.IN_PLACE_CONVERT: [
            ps_macros.CATG_PRESENCE,
            # ps_macros.IS_TARGET_STR,
            ps_macros.BINARY_CATG_PRESENCE,
            ps_macros.SMALL_CATG_PRESENCE,
            ps_macros.LARGE_CATG_PRESENCE,
        ],
        ps_macros.ONE_HOT: [
            ps_macros.CATG_PRESENCE,
            # ps_macros.IS_TARGET_STR,
            ps_macros.BINARY_CATG_PRESENCE,
            ps_macros.SMALL_CATG_PRESENCE,
            ps_macros.LARGE_CATG_PRESENCE,
        ],
        ps_macros.VECT: [ps_macros.TEXT_PRESENCE],
        ps_macros.MISSING: [ps_macros.MISSING_PRESENCE],
        ps_macros.CATG: [ps_macros.CATG_PRESENCE],
        ps_macros.SCALING: [
            ps_macros.NORMALIZED_MEAN,
            ps_macros.NORMALIZED_STD_DEV,
            ps_macros.NORMALIZED_VARIATION_ACROSS_COLUMNS,
        ],
        ps_macros.DATE: [ps_macros.DATE_PRESENCE],
        ps_macros.LEMMITIZE: [ps_macros.TEXT_PRESENCE],
        ps_macros.BALANCING: [ps_macros.IMBALANCE],
        ps_macros.LOG: [
            ps_macros.MAX_SKEW,
        ],
    }
    return selection_model[label]
