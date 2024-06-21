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

import libcst as cst
from libcst.metadata import ParentNodeProvider, PositionProvider


class SpecialVariableRetrieval(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        PositionProvider,
        ParentNodeProvider,
    )

    def __init__(self):
        self.as_names = {}

    def visit_Assign(self, node) -> Optional[bool]:
        assigned_target = node.targets[0]
        target = assigned_target.target
        check = hasattr(target, "value")
        if check and target.value not in self.as_names:
            value = node.value
            if isinstance(value, cst.Call):
                if hasattr(value.func, "attr"):
                    api = str(value.func.attr.value)
                else:
                    api = str(value.func.value)
                if api == "read_csv":
                    self.as_names["dataframe"] = target.value


class ReferenceRetrieval(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        PositionProvider,
        ParentNodeProvider,
    )

    def __init__(self):
        self.as_names = {}

    def visit_Assign(self, node) -> Optional[bool]:
        assigned_target = node.targets[0]
        target = assigned_target.target
        check = hasattr(target, "value")
        if check and target.value not in self.as_names:
            value = node.value
            if isinstance(value, cst.Call):
                pos = self.get_metadata(PositionProvider, node).start
                self.as_names[target.value] = pos.line


class ObjectRetrieval(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        PositionProvider,
        ParentNodeProvider,
    )

    def __init__(self, function_name):
        self.object = ""
        self.function_name = function_name

    def visit_Call(self, node) -> Optional[bool]:
        if hasattr(node.func, "attr"):
            if node.func.attr.value == self.function_name:
                object_name = node.func.value.value
                self.object = object_name


class StringRetrieval(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        PositionProvider,
        ParentNodeProvider,
    )

    def __init__(self):
        self.strings = set()

    def visit_SimpleString(self, node) -> Optional[bool]:
        string_value = node.value
        string_without_quote = string_value.replace("'", "")
        string_without_quote = string_without_quote.replace('"', "")
        self.strings.add(string_without_quote)


class PossibleColumnNamesApiMap(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        PositionProvider,
        ParentNodeProvider,
    )

    def __init__(self):
        self.column_name_api = {}

    def visit_SimpleString(self, node) -> Optional[bool]:
        string_value = node.value
        string_without_quote = string_value.replace("'", "")
        string_without_quote = string_without_quote.replace('"', "")
        api = "n/a"
        parent = self.get_metadata(ParentNodeProvider, node)

        while True:
            if (
                isinstance(parent, cst.Call)
                or isinstance(parent, cst.SimpleStatementLine)
                or isinstance(parent, cst.Module)
            ):
                break
            parent = self.get_metadata(ParentNodeProvider, parent)
        if isinstance(parent, cst.Call):
            func = parent.func
            if isinstance(func, cst.Name):
                api = func.value
            elif isinstance(func, cst.Attribute):
                api = func.attr.value
        if string_without_quote in self.column_name_api:
            api_list = self.column_name_api[string_without_quote]
            api_list.append(api)
        else:
            self.column_name_api[string_without_quote] = [api]


class ApiRetrieval(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        PositionProvider,
        ParentNodeProvider,
    )

    def __init__(self):
        self.object = ""
        self.function_names = {}

    def visit_Call(self, node) -> Optional[bool]:
        if hasattr(node.func, "attr"):
            api = str(node.func.attr.value)
        else:
            api = str(node.func.value)

        count = 0
        if api in self.function_names:
            count = self.function_names[api]
        self.function_names[api] = count + 1


class ImportMapRetrieval(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        PositionProvider,
        ParentNodeProvider,
    )

    def __init__(self):
        self.as_names = {}
        self.uses = {}

    def visit_AsName_name(self, node) -> None:
        pos = self.get_metadata(PositionProvider, node).start
        self.as_names[node.name.value] = pos.line

    def visit_ImportAlias(self, node) -> Optional[bool]:
        pos = self.get_metadata(PositionProvider, node).start
        self.as_names[node.name.value] = pos.line

    def visit_Name(self, node) -> Optional[bool]:
        if node.value in self.as_names:
            pos = self.get_metadata(PositionProvider, node).start
            if node.value not in self.uses:
                use_set = set()
            else:
                use_set = self.uses[node.value]
            use_set.add(pos.line)
            self.uses[node.value] = use_set


def get_import_map(code):
    source_tree = cst.parse_module(code)
    retriever = ImportMapRetrieval()
    wrapper = cst.metadata.MetadataWrapper(source_tree)
    wrapper.visit(retriever)
    line_2_import_map = {}
    for key1 in retriever.uses:
        uses = retriever.uses[key1]
        for use in uses:
            if use not in line_2_import_map:
                import_set = set()
            else:
                import_set = line_2_import_map[use]
            import_set.add(retriever.as_names[key1])
            line_2_import_map[use] = import_set

    return line_2_import_map


def get_definition_line_map(code):
    source_tree = cst.parse_module(code)
    retriever = ReferenceRetrieval()
    wrapper = cst.metadata.MetadataWrapper(source_tree)
    wrapper.visit(retriever)
    return retriever.as_names


def get_column_api_map(code):
    source_tree = cst.parse_module(code)
    retriever = PossibleColumnNamesApiMap()
    wrapper = cst.metadata.MetadataWrapper(source_tree)
    wrapper.visit(retriever)
    return retriever.column_name_api


def get_object_name(code, function_name):
    source_tree = cst.parse_module(code)
    retriever = ObjectRetrieval(function_name)
    wrapper = cst.metadata.MetadataWrapper(source_tree)
    wrapper.visit(retriever)
    return retriever.object


def get_api_count(code):
    source_tree = cst.parse_module(code)
    retriever = ApiRetrieval()
    wrapper = cst.metadata.MetadataWrapper(source_tree)
    wrapper.visit(retriever)
    function_names = retriever.function_names
    for api in function_names:
        if function_names[api] > 1:
            return "P"
    return "S"


def get_strings(code):
    source_tree = cst.parse_module(code)
    retriever = StringRetrieval()
    wrapper = cst.metadata.MetadataWrapper(source_tree)
    wrapper.visit(retriever)
    return retriever.strings


def get_special_variable(code):
    source_tree = cst.parse_module(code)
    retriever = SpecialVariableRetrieval()
    wrapper = cst.metadata.MetadataWrapper(source_tree)
    wrapper.visit(retriever)
    return retriever.as_names
