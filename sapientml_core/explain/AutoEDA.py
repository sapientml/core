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

import numpy as np
import pandas as pd
from sapientml.util.logging import setup_logger
from scipy import stats


class Dataset:
    """Dataset class."""

    def __init__(self, dataframe: pd.DataFrame, target_columns: list[str]):
        """__init__ method.

        Parameters
        ----------
        dataframe : pd.DataFrame
            dataframe input
        target_columns : list[str]
            Names of target columns.

        """
        self.data = dataframe
        self.target = target_columns


class EDA(Dataset):
    """EDA class."""

    def __init__(self, dataframe: pd.DataFrame, target_columns: list[str], ref="Dataset", log_level=1, logger=None):
        """__init__ method.

        Parameters
        ----------
        dataframe : pd.DataFrame
            dataframe input
        target_columns : list[str]
            Names of target columns.
        ref : Dataset
            Dataset class reference.

        """
        self.logger = logger
        if logger is None:
            self.logger = setup_logger()
        self.logger.info("AutoEDA is processing input dataset...")
        self.df = Dataset(dataframe, target_columns)
        self.log_level = log_level
        self.description = []
        self.ref = ref
        self.add_general_block()
        self.check_skewness()

    def __str__(self):
        lines = ""
        for i, j in self.description:
            comm = "#" if j == "markdown" else ""
            lines += comm + "\n".join(_ for _ in i) + "\n"
        return lines

    def check_skewness(self):
        """check_skewness method.

        skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean.

        """
        ref = self.ref
        skews = []
        for col in self.df.data:
            try:
                skew = self.df.data[col].skew()
            except TypeError:
                skew = 0
            if skew > 1 or skew < -1:
                skews.append([col, skew, "highly skewed"])
            elif skew > 0.5 or skew < -0.5:
                skews.append([col, skew, "moderately skewed"])

        if len(skews):
            skews = pd.DataFrame(skews, columns=["Column", "Skewness", "Skewness_Type"])

            self.description.append(
                (
                    [
                        "## Skewness",
                        "In probability theory and statistics, skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean. The skewness value can be positive, zero, negative, or undefined.",
                        "\nMore detail can be found [here](https://en.wikipedia.org/wiki/Skewness) and the [Probability and Statistics Tables and Formulae](http://tomlr.free.fr/Math%E9matiques/Math%20Complete/Probability%20and%20statistics/CRC%20-%20standard%20probability%20and%20Statistics%20tables%20and%20formulae%20-%20DANIEL%20ZWILLINGER.pdf) by Zwillinger and Kokoska.",
                        '\nHere are two samples Skewness data for positive and negative skew data.<img src="https://upload.wikimedia.org/wikipedia/commons/f/f8/Negative_and_positive_skew_diagrams_%28English%29.svg" alt="source:Wikimedia">',
                        f"\nFirst, we will calculate the skewness for each column in {ref} if each selected column has positive float/integer values.",
                        "\nSecond, we will review the following conditions.",
                        "\n- if $Skewness>1$ or $Skewness<-1$, we will consider it as highly skewed;",
                        "\n- if $0.5<Skewness<=1$ or $-0.5<Skewness<=-1$, we will consider it as moderate skewed;",
                        f"\n{skews.to_html()}",
                    ],
                    "markdown",
                )
            )

    def add_general_block(self):
        """add_general_block method.

        Adding markdown cells in jupyter notebook.

        """
        ref = self.ref
        if self.df.data is not None:
            n_rows = self.df.data.shape[0]
            n_col = self.df.data.shape[1]
            n_types = self.df.data.dtypes.unique()
            self.description.append(
                (
                    [
                        "# Exploratory Data Analysis (EDA)",
                        "## General Structure",
                        f"{ref} includes **{n_col}** columns and **{n_rows}** rows.",
                        f"There are **{len(n_types)}** different data types as follows: "
                        + f"*{u', '.join(str(_) for _ in n_types.tolist())}*.",
                    ],
                    "markdown",
                )
            )
            self.description.append(
                (
                    [
                        "Let's review the dataset description:",
                        f"{self.df.data.describe().T.to_html()}",
                    ],
                    "markdown",
                )
            )
            all_sum = self.df.data.isnull().sum()
            all_sum = all_sum[all_sum != 0]

            if len(all_sum):
                # generate content if there is a null value
                all_sum.sort_values(ascending=False, inplace=True)
                all_sum_filter = pd.DataFrame(
                    [(str(all_sum.index[i]), str(u)) for i, u in enumerate(all_sum[:20])], columns=["Column", "#Null"]
                )
                self.description.append(
                    (
                        [
                            f"**Is there any null value?** \nThe answer is **Yes**; let's review top {'20' if len(all_sum)>20 else str(len(all_sum))} of those columns with the number of Null values.",
                            f"\n{all_sum_filter.to_html()}",
                            f"\nAs partial of the results shown above, there are total **{len(all_sum)}** columns with Null values.",
                        ],
                        "markdown",
                    )
                )

            if self.df.target:
                if isinstance(self.df.target, list):
                    for col in self.df.target:
                        if col in self.df.data:
                            unique_vals = self.df.data[col].unique()
                            if len(unique_vals) < 11:
                                freq = pd.DataFrame(self.df.data[col].value_counts())
                                self.description.append(
                                    (
                                        [
                                            f"There is **{len(unique_vals)}** unique value in **{col}** column which is a target column.",
                                            "let's see frequency of values for the target column of {col}:",
                                            f"{freq.to_html()}",
                                        ],
                                        "markdown",
                                    )
                                )

    def cat_process(self, threshold=0.01, IQR_activation: bool = True, z_activation: bool = True):
        """cat_process method.

        Parameters
        ----------
        threshold : float
        IQR_activation : bool
            An interquartile range is a measure of statistical dispersion
            IQR = Q3 - Q1 (75th percentile - 25th percentile)
            True and otherwise False.
        z_activation : bool
            z-score is used to find the outliers.
            True and otherwise False.

        Returns
        -------
        hashmaps : dict
            A pivoted dataframe.
        df_desc : pd.DataFrame.
            describing a dataframe.

        """
        hashmaps = {}
        desc = []
        ratio = float(threshold / 100)
        for key in self.df.data:
            hashmaps = dict(self.df.data.pivot_table(columns=[key], aggfunc="size"))
            # skip all unique values &
            # skip all keys with single repatation (unique) &
            # skip large keys (100+) &
            # skip none values &
            if hashmaps:
                if (
                    len(hashmaps.keys()) != self.df.data[key].shape[0]
                    and sum(hashmaps.values()) != len(hashmaps.values())
                    and 1 < len(hashmaps.keys()) < 100
                    and len(hashmaps.values())
                ):
                    values = list(hashmaps.values())

                    # baseline
                    upper = np.quantile(values, 1 - ratio)
                    lower = np.quantile(values, ratio)

                    # IQR
                    Q1 = np.quantile(values, 0.25)
                    Q3 = np.quantile(values, 0.75)
                    IQR = Q3 - Q1

                    # Z-Score
                    z = np.abs(stats.zscore(values))

                    for index, cat in enumerate(hashmaps):
                        val = hashmaps[cat]
                        if (
                            val > upper
                            and (not IQR_activation or val > (Q3 + 1.5 * IQR))
                            and (not z_activation or z[index] > 3)
                        ):
                            desc.append((key, cat, hashmaps[cat], lower, upper, "Upper"))
                        elif (
                            val < lower
                            and (not IQR_activation or val < (Q1 - 1.5 * IQR))
                            and (not z_activation or z[index] > 3)  # we already have abs of z-score
                        ):
                            desc.append((key, cat, hashmaps[cat], lower, upper, "Lower"))
        df_desc = pd.DataFrame(desc, columns=["Field", "Value", "Frequency", "Lower", "Upper", "Criteria"])
        if len(df_desc):
            singular = "s" if len(df_desc) > 1 else ""
            singular_verb = "are" if len(df_desc) > 1 else "is"
            exp_statement = "> Upper(C_0)" if df_desc.iloc[0].Criteria == "Upper" else "< Lower(C_0)"
            self.description.append(
                (
                    [
                        "# Finding Intresting Datapoints",
                        "Let's process each field by their histogram frequency and check if there is any intresting data point.",
                        f"\nThere {singular_verb} **{len(df_desc)}** number of intresting value{singular} in the following column{singular}.",
                        "The below table shows each **Value** of each **Field**(column) with their total frequencies, **Lower** shows the lower frequency of normal distribution, **Upper** shows the upper bound frequency of normal distribution, and **Criteria** shows if the frequnecy passed **Upper bound** or **Lower bound**.",
                        f"{df_desc.to_html()}",
                        "\n\n",
                        f"For example, in the **{df_desc.iloc[0].Field}** column the value of **{df_desc.iloc[0].Value}** has **{df_desc.iloc[0].Frequency}** repeatation but this number is not between Lower bound({df_desc.iloc[0].Lower}) and Upper bound({df_desc.iloc[0].Upper}).\n\n",
                        f"Let     $C_0={df_desc.iloc[0].Value}$   and   $Freq(C_0)={df_desc.iloc[0].Frequency}$     ,   $Upper(C_0)={df_desc.iloc[0].Upper}$     ,   $Lower(C_0)={df_desc.iloc[0].Lower}$\n",
                        f"$Freq(C_0) {exp_statement}$.",
                    ],
                    "markdown",
                )
            )
        return hashmaps, df_desc

    def check_consistency(self, convert=False):
        """check_consistency method.

        Parameters
        ----------
        convert : bool
            convert bool string to numerical val.
            False and otherwise True.

        Returns
        -------
        pd.DataFrame
            converted dataframe if all the keys dtype is object.

        """
        for key in self.df.data:
            if self.df.data[key].dtype == "object":
                self.df.data[key] = self.__convert_object(self.df.data[key], convert, key)
        return self.df.data

    def __is_bool(self, unique_vals):
        """check if the given input can be considered as boolean"""
        if len(unique_vals) != 2:
            # total number of elements sould be two, to be considered
            return None
        # Check if all elements in unique_vals are strings
        if not all(isinstance(_, str) for _ in unique_vals):
            return None
        sorted_unique_vals = sorted([_.lower().strip() for _ in unique_vals])
        candidates = [
            # TODO: add dynamic values
            ["no", "yes"],
            ["false", "true"],
            ["incorrect", "correct"],
            ["0", "1"],
            ["n", "y"],
            ["f", "t"],
        ]

        for candidate in candidates:
            if sorted(candidate) == sorted_unique_vals:
                # return consistant index (i.e., yes=1 and no=0)
                return [candidate.index(_) for _ in sorted_unique_vals]
        return None

    def __convert_digit(self, cell, key):
        ref = self.ref
        try:
            cell = cell.astype(int)
        except ValueError:
            # if there is float value in list
            cell = cell.astype(float)
            if self.log_level >= 2:
                self.description.append(
                    (
                        [
                            f"Let's convert {ref}['{key}'] from Object to float because it includes at least a float values."
                        ],
                        "markdown",
                    )
                )
            self.description.append(([f"{ref}['{key}']={ref}['{key}'].astype(float)"], "code"))
        else:
            if self.log_level >= 2:
                self.description.append(
                    (
                        [f"Let's convert {ref}['{key}'] from Object to integer."],
                        "markdown",
                    )
                )
            self.description.append(([f"{ref}['{key}']={ref}['{key}'].astype(int)"], "code"))
        return cell

    def __convert_object(self, cell, convert: bool, key):
        """
        types source: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.isnumeric.html
        :params convert: convert bool string to numerical val
        :params key: the current title of column
        """
        ref = self.ref

        numeric = cell[pd.to_numeric(cell, errors="coerce").notnull()]
        #        alpha=cell[cell.str.isalpha()]
        # If cell is [True, nan], this is an object column, but str accessor raises an error.
        # fillna is used to prevent the error.
        alpha = cell[(cell.fillna("").str.isalpha()) | (cell.isna())]

        if len(alpha) == 0 and len(numeric) == len(cell):  # object but no alpha  # mixed of int/float
            return self.__convert_digit(cell, key) if convert else cell
        elif len(alpha) > 0:  # object: mixed of alpha and possible int/float
            unique_num = numeric.unique().tolist()
            unique_alpha = alpha.unique().tolist()

            bool_check = self.__is_bool(unique_alpha)

            if bool_check is not None and len(unique_num) > len(  # there is bool transformer
                unique_alpha
            ):  # numerical category is bigger
                # we have boolean values
                # replace values with bool_candidate indexes
                if self.log_level >= 1:
                    self.description.append(
                        (
                            [
                                f"{ref}['{key}'] includes string values of: {','.join(_ for _ in unique_alpha)}  where it is mixed with {len(unique_num)} different number of numeric values."
                            ],
                            "markdown",
                        )
                    )
                if convert:
                    for index, val in enumerate(unique_alpha):
                        cell = cell.replace(val, str(bool_check[index]))
                        if self.log_level >= 2:
                            self.description.append(
                                (
                                    [f"We can replace '{val}' with '{str(bool_check[index])}'"],
                                    "markdown",
                                )
                            )
                        self.description.append(
                            ([f"{ref}['{key}']={ref}['{key}'].replace('{val}',str({bool_check[index]}))"], "code")
                        )
                    # convert to target int or float
                    cell = self.__convert_digit(cell, key)
        return cell
