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
import pathlib
from collections import defaultdict
from typing import Optional

import libcst as cst
import pandas as pd
from libcst.metadata import ParentNodeProvider, PositionProvider
from sapientml.util.logging import setup_logger
from sapientml_core import internal_path
from sapientml_core.training.dataflowmodel import ast_operation as ast

logger = setup_logger()

sig2label = {}


class DependentApiCollector(cst.CSTVisitor):
    """DependentApiCollector class

    This script groups the APIs that are dependent on each other based on the given pipelins in the corpus.
    In SapientML, two APIs: A and B are dependent on each other if both A and B are applied on at least one column
    of the dataset in the given pipeline.

    """

    METADATA_DEPENDENCIES = (
        ParentNodeProvider,
        PositionProvider,
    )

    def __init__(self, relative_file_path, columns):
        self.as_names = {}
        self.columns = columns
        self.column2apis = {}
        self.user_defined_function = False
        self.relative_file_path = relative_file_path
        self.count = 0

    def visit_SimpleString(self, node: cst.SimpleString) -> Optional[bool]:
        """visit_SimpleString function

        Parameters
        ----------
        node : SimpleString
            It is libcst node class

        """
        string_value = node.value

        string_without_quote = string_value.replace("'", "")
        string_without_quote = string_without_quote.replace('"', "")

        if string_without_quote in self.columns:
            parent = self.get_metadata(ParentNodeProvider, node)
            line_number = -1
            while not (isinstance(parent, cst.SimpleStatementLine) or isinstance(parent, cst.Module)):
                if isinstance(parent, cst.Call):
                    pos = self.get_metadata(PositionProvider, parent).start
                    line_number = pos.line
                    break
                parent = self.get_metadata(ParentNodeProvider, parent)

            if line_number != -1:
                sig = self.relative_file_path + ":" + str(line_number)
                current_list = []
                if string_without_quote in self.column2apis.keys():
                    current_list = self.column2apis[string_without_quote]

                label = sig2label[sig] if sig in sig2label.keys() else "N/A"

                if label not in current_list and label != "N/A":
                    current_list.append(label)

                self.column2apis[string_without_quote] = current_list

    def visit_Assign(self, node: cst.Assign) -> Optional[bool]:
        """visit_Assign function

        Parameters
        ----------
        node : Assign
            It is libcst node class

        """
        try:
            assigned_target = node.targets[0]
            target = assigned_target.target
            check = hasattr(target, "value")
            if check and target.value not in self.as_names:
                value = node.value
                if isinstance(value, cst.Call):
                    func = value.func
                    if isinstance(func, cst.Name):
                        name = func.value
                    elif isinstance(func, cst.Attribute):
                        name = func.attr.value
                    self.as_names[target.value] = name
                else:
                    self.count += 1
                    self.as_names[target.value] = "@@VAR@@"
        except Exception:
            pass

    def visit_ImportAlias(self, node: cst.ImportAlias):
        """visit_ImportAlias function

        Parameters
        ----------
        node : ImportAlias
            It is libcst node class

        """
        if node.asname is not None:
            type_str = ""
            child = node.name
            while not isinstance(child, cst.Name):
                type_str = type_str + "_" + child.attr.value
                child = child.value
            type_str = type_str + "_" + child.value
            self.as_names[str(node.asname.name.value)] = str(type_str)

    def get_Call_info(self, node):
        """get_Call_info function

        Returns
        -------
        function_call : str

        """
        if isinstance(node.func, cst.Attribute):  # c.m(...)
            function_call = self.getCallFullyQualifiedName(node.func)
        elif isinstance(node.func, cst.Name):  # m(...)
            function_name = node.func.value
            function_call = function_name
        return function_call

    def getCallFullyQualifiedName(self, func_node):
        """getCallFullyQualifiedName function

        Returns
        -------
        str
            It returns fully qualified name

        """
        if isinstance(func_node, cst.Name):  # recursive call may trigger this condition
            return func_node.value
        if isinstance(func_node.value, cst.Name):
            module_name = func_node.value.value
            function_name = func_node.attr.value
            function_call = (
                (self.as_names[module_name] if module_name in self.as_names.keys() else module_name)
                + "."
                + function_name
            )
            return function_call
        elif isinstance(func_node.value, cst.Subscript):
            module_name = "UNKNOWN_TYPE"
            function_name = func_node.attr.value
            function_call = module_name + "." + function_name
            return function_call
        elif isinstance(func_node.value, cst.Call):
            function_name = func_node.attr.value
            return self.getCallFullyQualifiedName(func_node.value.func) + "." + function_name
        elif isinstance(func_node.value, cst.Attribute):
            function_name = func_node.attr.value
            if isinstance(func_node.value.value, cst.Name):
                return func_node.value.value.value + "." + function_name
            elif isinstance(func_node.value.value, cst.Call):
                attr_name = func_node.value.attr.value
                return (
                    self.getCallFullyQualifiedName(func_node.value.value.func) + "." + attr_name + "." + function_name
                )
            elif isinstance(func_node.value.value, cst.Subscript):
                attr_name = func_node.value.attr.value
                return "UNKNOWN_TYPE" + "." + attr_name + "." + function_name
            elif isinstance(func_node.value.value, cst.Attribute):
                return self.getAttributeFullyQualifiedName(func_node.value) + "." + function_name
            else:
                return "XXX"
        elif isinstance(func_node.value, cst.SimpleString):  # e.g., "".join()
            return "str." + func_node.attr.value
        elif isinstance(func_node.value, cst.Comparison):  # e.g., (z > 3).xxx()
            return "bool." + func_node.attr.value
        elif isinstance(func_node.value, cst.BinaryOperation):  # e.g., (a - b).min()
            return "bin_op." + func_node.attr.value
        else:
            return "XXX"

    def getAttributeFullyQualifiedName(self, func_node):
        """getAttributeFullyQualifiedName function

        Returns
        -------
        str
            It returns fully qualified attribute name.

        """
        function_name = func_node.attr.value
        if isinstance(func_node.value.value, cst.Name):
            attr_name = func_node.value.attr.value
            return func_node.value.value.value + "." + attr_name + "." + function_name
        elif isinstance(func_node.value.value, cst.Call):
            attr_name = func_node.value.attr.value
            return self.getCallFullyQualifiedName(func_node.value.value.func) + "." + attr_name + "." + function_name
        elif isinstance(func_node.value.value, cst.Subscript):
            attr_name = func_node.value.attr.value
            return "UNKNOWN_TYPE" + "." + attr_name + "." + function_name
        elif isinstance(func_node.value.value, cst.Attribute):
            attr_name = func_node.value.attr.value
            return self.getAttributeFullyQualifiedName(func_node.value) + "." + attr_name + "." + function_name
        else:
            return "XXX"


def find_column_to_apis(file_path, dataset_path):
    """Extracting API's used on columns.

    Parameters
    ----------
    file_path : str
        It contains file path in string format.
    dataset_path : PosixPath
        It contains path from pathlib.PosixPath class.

    Returns
    -------
    dict
        It returns collector.column2apis in dictionary format.

    """
    source_tree = ast.construct_tree(file_path)
    data = pd.read_csv(dataset_path, encoding="ISO-8859-1")
    parts = file_path.split("/")
    relative_file_path = parts[-1]
    collector = DependentApiCollector(relative_file_path, data.columns)
    wrapper = cst.metadata.MetadataWrapper(source_tree)
    wrapper.visit(collector)
    return collector.column2apis


def get_dataset_folder_name(file_path):
    notebook_info_path = file_path.with_suffix(".info.json")
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


def get_dataset_path(file_path):
    """Fetches the dataset path for the given pipeline.

    Parameters
    ----------
    file_path : PosixPath
        It contains path from pathlib.PosixPath

    Returns
    -------
    target_csv_file : PosixPath
        It contains path of the target_csv_file from pathlib.PosixPath class

    """
    dataset_folder_name = get_dataset_folder_name(file_path)
    parent = file_path.parent / dataset_folder_name
    csv_file_lists = pathlib.Path(str(parent).replace(internal_path.clean_notebooks_dir_name, "dataset")).rglob("*.csv")
    target_csv_file = None
    for csv_file in csv_file_lists:
        target_csv_file = csv_file
    return target_csv_file


def main():
    """Extract API's from pipelines and create dependent API's map."""
    annotated_notebooks = internal_path.project_labels_path

    label_data = pd.read_csv(annotated_notebooks, encoding="utf-8", usecols=["file_name", "new_label", "line_number"])

    for _, row in label_data.iterrows():
        key = str(row["file_name"]) + ":" + str(row["line_number"])
        new_label = str(row["new_label"])
        if new_label.startswith("PREPROCESS"):
            sig2label[key] = new_label.split("#")[0]

    final_dependency_list = defaultdict(int)

    path_list = pathlib.Path(internal_path.clean_dir).rglob("*.py")
    for path in path_list:
        logger.info(f"FILE:{path}")
        try:
            file_path = str(path)
            dataset_path = get_dataset_path(path)
            col2apis = find_column_to_apis(file_path, dataset_path)
            logger.info(col2apis)
            for key in col2apis:
                if len(col2apis[key]) > 1:
                    new_key = str(col2apis[key])
                    final_dependency_list[new_key] += 1
        except Exception:
            import traceback

            traceback.print_exc()

    sorted_map = {k: v for k, v in sorted(final_dependency_list.items(), key=lambda item: item[1], reverse=True)}

    with open(internal_path.training_cache / "dependent_labels.json", "w", encoding="utf-8") as outfile:
        json.dump(sorted_map, outfile, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="Tag for output files and dirs.")
    args = parser.parse_args()
    if args.tag:
        internal_path.training_cache = internal_path.training_cache / args.tag

    main()
