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

from typing import Optional

import pandas as pd


class Pipeline_Explanation:
    """Pipeline_Explanation class.

    Attributes
    ----------
    Feature_exp_map : dict
        explanation of features.
    operators : dict
        dictionary of operator symbols with their corresponding names.

    """

    Feature_exp_map = {
        "feature:missing_values_presence": "There are missing values among the columns with a score of **{actual_value}** which is {operator} **{threshold}**.",
        "feature:max_normalized_stddev": "The maximum of standard deviation of the normalized data (among the columns) is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:target_imbalance_score": "The imbalance score among target values of the columns, is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:str_category_presence": "The category score of the columns is present with value of **{actual_value}** which is {operator} **{threshold}**.",
        "feature:str_category_binary_presence": "There is a binary category among the columns with a score of **{actual_value}** which is {operator} **{threshold}**.",
        "feature:str_category_small_presence": "There is a small number of categories (between 2 to 5) among the columns with a score of **{actual_value}** which is {operator} **{threshold}**.",
        "feature:str_text_presence": "There is a text value that includes blank space among the columns with a score of **{actual_value}** which is {operator} **{threshold}**.",
        "feature:str_date_presence": "The date presents among the columns with a score of **{actual_value}** which is {operator} {threshold}.",
        "feature:str_category_large_presence": "There is a large number of categories (between 2 to 5) among the columns with a score of **{actual_value}** which is {operator} **{threshold}**.",
        "feature:max_normalized_mean": "The average of normalized values of the columns by considering only ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] type is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:normalized_variation_across_columns": "The average of normlized values of the columns by considering only ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] type is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:max_skewness": "The maximum of skewness (according to scipy.stats.skew) among the columns is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:not_basic_cols": "The number of type of columns are out of regular types (int, float, string, boolean) is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:dominant_0.8": "The number of top relative normalized frequency of values (dividing all values by the sum of values) with greater than 0.8 is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:dominant_0.9": "The number of top relative normalized frequency of values (dividing all values by the sum of values) with greater than 0.9 is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:missing_10-50_cols": "The ratio of missing values among the columns are between 1% to 10% with the number of **{actual_value}** which is {operator} **{threshold}**.",
        "feature:missing_50-90_cols": "The ratio of missing values among the columns are between 50% to 90% with the number of **{actual_value}** which is {operator} **{threshold}**.",
        "feature:missing_90-100_cols": "The ratio of missing values among the columns are between 90% to 100% with the number of **{actual_value}** which is {operator} **{threshold}**.",
        "feature:missing_0-10_cols": "The ratio of missing values among the columns are between 0% to 10% with the number of **{actual_value}** which is {operator} **{threshold}**.",
        "feature:missing_10-50_cols": "The ratio of missing values among the columns are between 1% to 10% with the number of **{actual_value}** which is {operator} **{threshold}**.",
        "feature:missing_10-50_cols": "The ratio of missing values among the columns are between 1% to 10% with the number of **{actual_value}** which is {operator} **{threshold}**.",
        "feature:negative_cols": "The number of negative values among numerical columns is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:median_0-1_cols": "The total number of median of the columns with less than 1 is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:median_1-10_cols": "The total number of median of the columns between 1 to 10, is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:median_10-100_cols": "The total number of median of the columns between 10 to 100, is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:median_100-1000_cols": "The total number of median of the columns between 100 to 1,000, is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:median_1000-10000_cols": "The total number of median of the columns between 1,000 to 10,000, is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:median_10000_cols": "The total number of median of the columns over 10,000, is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:iqr_0-1_cols": "The total number of values with interquartile range (IQR)<=1 is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:iqr_1-10_cols": "The total number of values with 1<interquartile range (IQR)<=10 is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:iqr_10-100_cols": "The total number of values with 10<interquartile range (IQR)<=100 is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:iqr_100-1000_cols": "The total number of values with 100<interquartile range (IQR)<=1,000 is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:iqr_1000-10000_cols": "The total number of values with 1000<interquartile range (IQR)<=10,000 is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:iqr_10000_cols": "The total number of values with interquartile range (IQR)>10,000 is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:str_date": "The total number of columns with date type (at least 80% of values has a date type) is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:str_catg": "The number of category column is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:str_num": "The number of numerical column (where at least 90% of values are numerical) is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:str_other": "The number of nonnumerical and noncategorical column is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:target_str_catg": "The number of target string category colums, is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:target_str_text": "The number of target text columns, is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:target_str_other": "The number of nonnumerical and noncategorical columns, is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:target_catg_num_max": "The maximum number of number of categorical columns, is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:target_catg_num_min": "The minimum number of number of categorical columns, is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:ttest_max": "The maximum number of identical columns (where T-Test of feature column > 0.01 against target columns), is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:ttest_min": "The minimum number of identical columns (where T-Test of feature column > 0.01 against target columns), is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:kstest_max": "The maximum number of identical distributions columns (where Kolmogorov-Smirnov test of feature column > 0.01 against target columns), is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:kstest_min": "The minimum number of identical distributions columns (where Kolmogorov-Smirnov test of feature column > 0.01 against target columns), is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:pearsonr_corr_max": "The maximum number of correlated columns against the target columns (where the Pearson correlation coefficient > 0.6), is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:pearsonr_corr_min": "The minimum number of correlated columns against the target columns (where the Pearson correlation coefficient > 0.6), is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:pearsonr_p_max": "The maximum number of correlated columns against the target columns (where the Pearson correlation p-value > 0.01), is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:pearsonr_p_min": "The minimum number of correlated columns against the target columns (where the Pearson correlation o-value > 0.01), is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:target_kurtosis_max": "The maximum value of target kurtosis is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:target_kurtosis_min": "The minimum value of target kurtosis is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:missing_values_special": 'The total number of special values that include ["?", "??", "-", "--", "---"] is **{actual_value}** which is {operator} **{threshold}**.',
        "feature:missing_values_rows": "The total number of missing values, is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:missing_values_named_columns": "The total number of columns with a title is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:avg_num_words": "The average number of words is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:num_unique_words": "The average number of unique words is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:str_catg_small": "The total number of small categories (the number of unique values < 20) is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:str_catg_large": "The total number of large categories (more than 80% unique values) is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:num_catg_binary": "The total number of binary categories (the number of unique values = 2) is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:num_catg_small": "The total number of small numerical categories (the number of unique values < 20) is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:num_catg_large": "The total number of large categories (more than 80% unique values) is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:str_text": "The ratio of text columns is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:num_catg": "The number of category columns is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:num_cont": "The number of columns with continues values is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:str_catg_binary": "The number of binary categories is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:outlier_cols": "The number of outlier columns is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:many_outlier_cols": "The ratio of outlier columns is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:kurtosis_normal": "The total number of columns with normal Kurtosis (kurtosis<0.5) is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:kurtosis_uniform": "The total number of columns with normal Kurtosis (kurtosis<-1.0) is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:kurtosis_tailed": "The total number of columns with uniform Kurtosis (kurtosis>=0.5) is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:dist_normal": "The total number of columns with normal distribution (Kolmogorov-Smirnov test of feature with normal distribution < 0.5) is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:dist_uniform": "The total number of columns with uniform distribution (Kolmogorov-Smirnov test of feature with uniform distribution < -1.0) is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:dist_poisson": "The total number of columns with poisson distribution (Kolmogorov-Smirnov test of feature with poisson distribution >= 0.5) is **{actual_value}** which is {operator} **{threshold}**.",
        "correlated_cols": "The total number of correlated columns is **{actual_value}** which is {operator} **{threshold}**.",
        "feature:num_of_rows": "The total number of observations (rows) is **{actual_value}** which is {operator} **{threshold}**.",
    }

    Operators = {
        ">": "greater than",
        "<": "smaller than",
        "=": "equal to",
        "<=": "smaller or equal to",
        ">=": "greater or equal to",
    }

    def __init__(self, skeleton: dict, explanation: dict, run_info: Optional[dict] = None):
        """__init__ method.

        Parameters
        ----------
        skeleton : dict
            Probabilty score and other details of preprocess and model components.
        explanation : dict
            pipeline explanation information.
        run_info : dict, optional
            execution results, logs and other information.

        """
        self.skeleton = skeleton
        self.explanation = explanation
        self.run_time = run_info

    def get_feature_explanation(self, action, cols, probability):
        """get_feature_explanation method.

        Parameters
        ----------
        action : str
            Name of the preprocess component.
        cols : List[str]
            list of the column names.
        probability : float
            Number used to determine the confidence level.

        Returns
        -------
        str

        """
        explain = []
        counter = 0
        explain.append(
            "## ML Explainability Tips:\n\nThe following codes are generated with a confidence level of **{probability}** because of the following reasons apply to **{len_cols}** column(s) that include {cols}.\n\n".format(
                cols=cols, len_cols=len(cols), probability=probability
            )
        )
        for predicate in self.skeleton[action]["predicates"]:
            feature = predicate["feature_name"]
            operator = self.Operators[predicate["operator"]]
            threshold = predicate["threshold"]
            actual_value = predicate["actual_value"]
            counter += 1
            explain.append(
                "1. "
                + self.Feature_exp_map[feature].format(
                    cols=cols, operator=operator, threshold=round(threshold, 3), actual_value=round(actual_value, 3)
                )
            )
        return "\n".join(_ for _ in explain)

    @staticmethod
    def get_model_exp(model):
        """get_model_exp method.

        Parameters
        ----------
        model : str
            Model Name.

         Returns
         -------
         str
             explanation of the model.

        """
        model_expl = ""
        if model:
            if model.startswith("MODEL"):
                model_type = model.split(":")[1]
                model_name = model.split(":")[2]
                model_expl = f"## ML Explainability Tips:\n\nThis is a **{model_type}** problem and we are selecting **{model_name}** "
        return model_expl

    def process(self):
        """process method.

        Returns
        -------
        explains : List[dict]
            Detailed explanation about best model.

        """
        explains = []
        ds = []
        for section in self.explanation:
            for action in self.explanation[section]:
                if "explanation" in self.explanation[section][action]:
                    code = self.explanation[section][action]["code"]
                    explain = self.explanation[section][action]["explanation"]
                    # meta = explain["relevant_meta_feature_list"]
                    probability = round(self.skeleton[action]["probability"], 2)
                    cols = explain["relevant_column_list"]
                    explains.append(
                        {
                            "code": code,
                            "action": action,
                            "explanation": self.get_feature_explanation(action, cols, probability),
                            "added": False,
                        }
                    )
            if section == "model":
                if "target_component_name" in self.explanation[section][action]:
                    model = self.explanation[section][action]["target_component_name"]
                    cols = self.explanation[section][action]["relevant_meta_feature_list"]
                    model_code = self.explanation[section]["code"]
                    prob = round(self.skeleton[model] / 2, 2)
                    explain_model = (
                        self.get_model_exp(model)
                        + f"with a probability of **{prob}**. The following are alternative options.\n"
                    )

                    metric = None  # in case run_time.json is not available
                    best_model_score = "N/A"
                    for key in self.skeleton.keys():
                        val = self.skeleton[key]
                        current_model = key.split(":")[2]
                        if key.startswith("MODEL:"):
                            score = "N/A"

                            if self.run_time:
                                for k in self.run_time:
                                    code_exec = self.run_time[k]
                                    if current_model.lower() in code_exec["content"]["model"]["label_name"].lower():
                                        if code_exec["run_info"]["score"]:
                                            score = round(code_exec["run_info"]["score"], 3)
                                            metric = code_exec["run_info"]["metric"]

                                        if model == key:  # top performed model
                                            best_model_score = score
                            if metric:
                                ds.append([current_model, f"{round(val/2,2)}", score])
                            else:
                                ds.append([current_model, f"{round(val/2,2)}"])
                    if metric:
                        ds_df = pd.DataFrame(ds, columns=["Model", "Probability", metric])
                    else:
                        ds_df = pd.DataFrame(ds, columns=["Model", "Probability"])
                    if ds_df.shape[0]:
                        ds_df.index = ds_df.index + 1
                        cols = []
                        explains.append(
                            {
                                "code": [model_code],
                                "action": model,
                                "explanation": explain_model  # basic_explain+
                                + (
                                    f"This model is selected based on the best **{metric}** of **{best_model_score}**.\n"
                                    if metric
                                    else ""
                                )
                                + ds_df.to_html(),
                                "added": False,
                            }
                        )
        return explains
