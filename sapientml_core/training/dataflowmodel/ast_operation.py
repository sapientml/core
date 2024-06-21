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

from typing import Optional, Union

import libcst as cst
import sapientml.macros as macros
from libcst import RemoveFromParent
from libcst.metadata import ParentNodeProvider, PositionProvider


class NameTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (
        ParentNodeProvider,
        PositionProvider,
    )

    def __init__(self, replacement):
        self.as_names = {}
        self.count = 0
        self.replacement = replacement

    def leave_Name(self, original_node, updated_node) -> cst.CSTNode:
        source_string = original_node.value
        if source_string in self.replacement.keys():
            return updated_node.with_changes(value=self.replacement[source_string])
        else:
            return original_node

    def leave_SimpleString(self, original_node: cst.Name, updated_node: cst.Name) -> cst.CSTNode:
        source_string = original_node.value
        if source_string in self.replacement.keys():
            return updated_node.with_changes(value='"' + self.replacement[source_string] + '"')
        else:
            return original_node

    def get_LineNumber(self, node):
        pos = self.get_metadata(PositionProvider, node).start
        return pos.line


class ArgumentRemover(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (
        ParentNodeProvider,
        PositionProvider,
    )

    def __init__(self, model_name):
        self.target = ""
        self.model_name = model_name

    def leave_Arg(self, original_node: cst.Arg, updated_node: cst.Arg) -> Union[cst.Arg, cst.RemovalSentinel]:
        parent = self.get_metadata(ParentNodeProvider, original_node)
        while not isinstance(parent, cst.Call):
            parent = self.get_metadata(ParentNodeProvider, parent)

        func = parent.func
        name = None
        if isinstance(func, cst.Name):
            name = func.value
        elif isinstance(func, cst.Attribute):
            name = func.attr.value
        if name == self.model_name:
            return RemoveFromParent()
        return updated_node


class ModelTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (
        ParentNodeProvider,
        PositionProvider,
    )

    def __init__(self, model_name):
        self.target = ""
        self.model_name = model_name

    def visit_Assign(self, node) -> Optional[bool]:
        assigned_target = node.targets[0]
        target = assigned_target.target
        check = hasattr(target, "value")
        if check:
            value = node.value
            if isinstance(value, cst.Call):
                func = value.func
                name = None
                if isinstance(func, cst.Name):
                    name = func.value
                elif isinstance(func, cst.Attribute):
                    name = func.attr.value
                if name == self.model_name:
                    self.target = target.value


def transform_model_code(source_code, model_label, metric=None):
    source_tree = cst.parse_module(source_code)
    model_name = model_label.split(":")[2]
    transformer = ModelTransformer(model_name)
    wrapper = cst.metadata.MetadataWrapper(source_tree)
    modified_tree = wrapper.visit(transformer)
    code = modified_tree.code.splitlines()
    if metric == macros.Metric.AUC or metric == macros.Metric.Gini:
        transformed_code = (
            code[0]
            + "\n"
            + transformer.target
            + ".fit(__feature_train, __target_train)\n__y_pred = "
            + transformer.target
            + ".predict_proba(__feature_test)"
        )
    else:
        transformed_code = (
            code[0]
            + "\n"
            + transformer.target
            + ".fit(__feature_train, __target_train)\n__y_pred = "
            + transformer.target
            + ".predict(__feature_test)"
        )
    return transformed_code


def remove_arguments(source_code, model_name):
    source_tree = cst.parse_module(source_code)
    transformer = ArgumentRemover(model_name)
    wrapper = cst.metadata.MetadataWrapper(source_tree)
    modified_tree = wrapper.visit(transformer)
    return modified_tree.code


def replaceString(source_tree, replacement):
    transformer = NameTransformer(replacement)
    wrapper = cst.metadata.MetadataWrapper(source_tree)
    modified_tree = wrapper.visit(transformer)
    return modified_tree


def construct_tree(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as file:
        code_content = file.read()
        parts = code_content.split("### Evaluation Template: ")
        code_content = parts[0]
        source_tree = cst.parse_module(code_content)
    return source_tree
