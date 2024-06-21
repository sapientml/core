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

import ast
import os
import subprocess
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

from sapientml.util.logging import setup_logger
from sapientml_core import internal_path
from sapientml_core.training import project_corpus

logger = setup_logger()


def catch_names(node):
    if isinstance(node, list):
        ret = set()
        for n in node:
            ret = ret.union(catch_names(n))
        return ret
    elif isinstance(node, ast.AST):
        ret = set()
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Name):
                ret.add(subnode)
        return ret
    else:
        return {}


def catch_first_name(assign_tgt):
    if isinstance(assign_tgt, ast.Name):
        return assign_tgt
    elif isinstance(assign_tgt, ast.Attribute):
        return catch_first_name(assign_tgt.value)
    elif isinstance(assign_tgt, ast.Subscript):
        real_target = assign_tgt
        while isinstance(real_target, ast.Subscript):
            real_target = real_target.value
        return catch_first_name(real_target)
    elif isinstance(assign_tgt, ast.Tuple) or isinstance(assign_tgt, ast.List):
        return catch_first_name(assign_tgt.elts[0])
    else:
        raise Exception(f"Unsupport type {type(assign_tgt)}")


def get_method_name(call_node):
    if isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Attribute):
        return call_node.func.attr
    else:
        return ""


class Injector(ast.NodeTransformer):
    """
    Instruments the pipelines with code snippets to collect snapshots of dataset.
    """

    def __init__(self):
        self.training_stmt_pattern = ["Dataset", "fit", "OLS", "DMatrix"]
        self.trace_vars = set()
        self.last_train_lineno = None

    def run(self, source_code_after_fixing_csv_path, script_path, new_script_path):
        tree = ast.parse(source=source_code_after_fixing_csv_path, filename=script_path)

        tree = self.visit(tree)

        loading_statements = ast.parse(
            f"""
with open("{new_script_path.replace('.py', '.txt')}", "w") as f:
    f.write('started')
import types
import importlib.machinery
loader = importlib.machinery.SourceFileLoader('df_collector', '{str(Path(__file__).parent / "df_collector.py")}')
mod = types.ModuleType(loader.name)
loader.exec_module(mod)
update_column_names = mod.update_column_names
collector = dict()
"""
        ).body

        # dumping statement
        # also dump the line no. of our determined training stmt
        dumping_statement = ast.parse(
            f"""
collector = [collector, {self.last_train_lineno}]
import json
with open("{new_script_path.replace('.py', '.json')}", 'w', encoding='utf-8') as f: json.dump(collector, f, indent=4)
"""
        ).body

        tree.body = loading_statements + tree.body + dumping_statement

        return ast.unparse(tree)

    def visit_Call(self, node):
        if self.last_train_lineno is None and get_method_name(node) in self.training_stmt_pattern:
            self.last_train_lineno = node.lineno
        return node

    def visit_Assign(self, node):
        if self.last_train_lineno and node.lineno >= self.last_train_lineno:
            return node
        target = catch_first_name(node.targets[0])
        var_out = catch_names(node.targets)
        var_in = catch_names(node.value)

        # debug
        if isinstance(target, ast.Subscript):
            logger.debug(node.lineno)
            logger.debug(ast.dump(node.targets[0]))

        target = target.id
        var_out = set([item.id for item in var_out if isinstance(item, ast.Name)])
        var_in = set([item.id for item in var_in if isinstance(item, ast.Name)])

        if get_method_name(node.value) == "read_csv":
            self.trace_vars.add(target)

        if any([item in self.trace_vars for item in var_in]):
            for item in var_out:
                self.trace_vars.add(item)

        if target in self.trace_vars:
            return [
                node,
                ast.parse(f"update_column_names(collector, {node.lineno}, {target}, '{target}')"),
            ]
        return node


def instrument_pipeline(pipeline, new_path):
    """Instruments a given pipeline with code snippets to collect snapshots of dataset using Injector.

    Parameters
    ----------
    pipeline : str
        pipeline path
    new_path : str
        new_path of dataset
    """
    with open(pipeline, "r", encoding="utf-8") as f:
        source_code = f.read()
    source_code_after_fixing_csv_path = source_code.replace("../../dataset", str(internal_path.corpus_path / "dataset"))
    source_code_after_injecting_targeted_lines = Injector().run(source_code_after_fixing_csv_path, pipeline, new_path)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    with open(new_path, "w", encoding="utf-8") as f:
        f.write(source_code_after_injecting_targeted_lines)


def run(file_path):
    """Executes the instrumented version of the pipeline to store the snapshots of the dataset.

    Parameters
    ----------
    file_path : str

    Returns
    -------
    int
       proccess of returncode

    """
    process = subprocess.Popen(
        "python" + " " + file_path,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output, err = process.communicate()
    output_content = output.decode("utf-8")
    error = err.decode("utf-8")
    logger.info(f"output_content:{output_content}")
    logger.error(f"error:{error}")
    return process.returncode


def dump_final_dataframe(project):
    """This function Validating  dataset_pkl  and replacing .py file with col_dumper file.

    Parameters
    ----------
    project : ProjectInfo
        The parameter has details information of pipleline.

    Returns
    -------
    None
       if file_path not exit.

    Raises
    ------
    AssertionError
        This will give error when assert returncode == 0.

    """
    pipeline_path = project.pipeline_path
    project_root = internal_path.clean_dir
    pipeline_relative_path = project.pipeline_path[len(str(project_root)) + 1 :]
    new_script_path = str(internal_path.training_cache / "dataset-snapshots" / pipeline_relative_path).replace(
        ".py", "_col_dumper.py"
    )
    if os.path.exists(new_script_path.replace(".py", ".json")):
        return None
    instrument_pipeline(pipeline_path, new_script_path)

    dataset_pkl = new_script_path.replace(".py", ".pkl")
    if os.path.exists(dataset_pkl):
        return None

    stime = time.time()
    returncode = run(new_script_path)
    ttime = time.time()
    logger.info(f"TIME SPENT:, {ttime - stime}")
    assert returncode == 0


def _run(data):
    total_number_target_pipelines, projects, i = data
    logger.info(f"RUNNING:, {i + 1}, out of, {total_number_target_pipelines}, PIPELINE: {projects[i].pipeline_path}")
    try:
        dump_final_dataframe(projects[i])
    except Exception:
        import traceback

        traceback.print_exc()


def main():
    """This main function getting all the pipeline details and parse the pipelines."""
    corpus = project_corpus.ProjectCorpus()
    projects = corpus.project_list

    total_number_target_pipelines = len(projects)

    if not os.path.exists(internal_path.training_cache / "dataset-snapshots"):
        os.makedirs(internal_path.training_cache / "dataset-snapshots")

    data = []
    for i in range(0, total_number_target_pipelines):
        data.append((total_number_target_pipelines, projects, i))
    p = Pool(cpu_count())
    values = [item for item in data]
    p.map(_run, values)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="Tag for output files and dirs.")
    args = parser.parse_args()
    if args.tag:
        internal_path.training_cache = internal_path.training_cache / args.tag
    main()
