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


import logging
import os
import shutil
import subprocess
import time
from multiprocessing import Pool, cpu_count

from sapientml.util.logging import setup_logger
from sapientml_core import internal_path
from sapientml_core.training import project_corpus
from sapientml_core.training.augmentation import mutator
from tqdm import tqdm

TIME_LIMIT = "10m"
SKIP_GOOD_RUN = True
SKIP_EXECED_ORIGINAL = False
SKIP_TLE = False
REGEN_MUTATION = True
GEN_DF_DUMPER = True

# whether to investigate notebooks other than the best notebook
run_all_notebooks = False

run_original = False
run_all_types = True
run_df_dumper = False

logger = setup_logger()


def execute(
    pipeline_name_without_extension,
    modelname,
    script_path,
    work_dir=None,
    skip_good_run=True,
    skip_timeout=True,
):
    """Executes the pipelines and update the results.

    Parameters
    ----------
    pipeline_name_without_extension : str
        It returns a string variable
    modelname : str
        It returns a string variable
    script_path : PosixPath
        It contains path from pathlib.PosixPath
    work_dir : PosixPath
        It contains path from pathlib.PosixPath
    skip_good_run : bool
        True and otherwise False
    skip_timeout : bool
        True and otherwise False

    """
    run_output_dir = internal_path.execution_cache_dir / pipeline_name_without_extension / modelname
    if not os.path.exists(run_output_dir):
        os.makedirs(run_output_dir)

    if os.path.exists(run_output_dir / "returncode.txt"):
        with open(run_output_dir / "returncode.txt", "r", encoding="utf-8") as f:
            code = f.read()
        code = int(code.strip())
        if code == 0 and skip_good_run:
            with open(os.path.join(run_output_dir, "stdout.txt"), "r", encoding="utf-8") as f:
                stdout = f.read()
            stdout = stdout.strip()
            stdout = stdout if len(stdout) <= 1000 else stdout[-1000:]
            logger.info(stdout)
            logger.info("a good run, skip")
            return
        if code == 124 and skip_timeout:
            logger.info("previously timeout, skip")
            return

    stime = time.time()

    subproc = subprocess.Popen(
        f"timeout {TIME_LIMIT} python " + str(script_path).replace(" ", "\\ "),
        cwd=work_dir,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout, stderr = subproc.communicate()
    ttime = time.time()
    if not os.path.exists(run_output_dir):
        os.makedirs(run_output_dir)
    with open(os.path.join(run_output_dir, "stdout.txt"), "w", encoding="utf-8") as f:
        f.write(stdout)
    with open(os.path.join(run_output_dir, "stderr.txt"), "w", encoding="utf-8") as f:
        f.write(stderr)
    with open(os.path.join(run_output_dir, "returncode.txt"), "w", encoding="utf-8") as f:
        f.write(str(subproc.returncode))
    with open(os.path.join(run_output_dir, "time.txt"), "w", encoding="utf-8") as f:
        f.write(str(ttime - stime))
    if subproc.returncode != 0:
        logger.warning(
            f"!!! Error encountered @ {pipeline_name_without_extension}/{modelname}: returncode = {subproc.returncode}"
        )

    else:
        stdout = stdout.strip()
        stdout = stdout if len(stdout) <= 1000 else stdout[-1000:]


def run(args):
    """Wrapper function for execute() function.

    Parameters
    ----------
    args : tuple

    Returns
    -------
    j : int
    now_t : str

    """
    j, now_t, pipeline_name_without_extension = args
    execute(
        pipeline_name_without_extension,
        now_t,
        internal_path.execution_cache_dir / pipeline_name_without_extension / now_t / "script.py",
        internal_path.clean_dir,
        SKIP_GOOD_RUN,
    )
    return j, now_t


def generate_mutated_accuracy(project):
    """Executes original and mutated pipelines.

    Parameters
    ----------
    project : ProjectInfo
        It contains ProjectInfo class details

    """
    pipeline_name_without_extension = project.notebook_name

    with open(project.pipeline_path, "r", encoding="utf-8") as f:
        original_script = f.read()

    original_script = original_script.replace("../../dataset", str(internal_path.corpus_path / "dataset"))

    script_path = internal_path.execution_cache_dir / pipeline_name_without_extension / "original"
    if not os.path.exists(script_path):
        os.makedirs(script_path)
    script_path = script_path / "script.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(original_script)

    # Running Original Pipeline
    execute(
        pipeline_name_without_extension,
        "original",
        script_path,
        internal_path.clean_dir,
        SKIP_GOOD_RUN,
    )

    origin_type, muted_types = "", []
    try:
        origin_type, muted_types = mutator.model_mutation(
            script_path,
            internal_path.execution_cache_dir / pipeline_name_without_extension,
            REGEN_MUTATION,
            GEN_DF_DUMPER,
        )
    except Exception:
        pass

    if run_all_types:
        exec_trace_files = ["stdout.txt", "stderr.txt", "time.txt", "returncode.txt"]
        if SKIP_EXECED_ORIGINAL and all(
            [
                os.path.exists(internal_path.execution_cache_dir / pipeline_name_without_extension / "original" / item)
                for item in exec_trace_files
            ]
        ):
            # detected that the original is already executed and allow to skip
            for file in exec_trace_files:
                shutil.copy(
                    internal_path.execution_cache_dir / pipeline_name_without_extension / "original" / file,
                    internal_path.execution_cache_dir / pipeline_name_without_extension / origin_type / file,
                )
            pass
        else:
            # run original
            script_path = (
                internal_path.execution_cache_dir / pipeline_name_without_extension / origin_type / "script.py"
            )
            if os.path.exists(script_path):
                execute(
                    pipeline_name_without_extension,
                    origin_type,
                    script_path,
                    internal_path.clean_dir,
                    SKIP_GOOD_RUN,
                )

        muted_types = [(j, now_t, pipeline_name_without_extension) for j, now_t in enumerate(muted_types)]

        # then run muted types
        with Pool(cpu_count() // 4 or 1) as p:
            for j, now_t in tqdm(p.imap(run, muted_types)):
                logger.info(f"  [{j + 1}/{len(muted_types) + 1}] run type {now_t}")
        p.join()


def main(n, k):
    """Creates augmented pipelines and executes it.

    Parameters
    ----------
    n : int
    k : int

    """
    corpus = project_corpus.ProjectCorpus()
    if not os.path.exists(internal_path.execution_cache_dir):
        os.makedirs(internal_path.execution_cache_dir)
    logging.basicConfig(
        filename=internal_path.execution_cache_dir / f"mutation_{k}.log",
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for i, project in enumerate(corpus.project_list):
        if i % n != k:
            continue
        logger.info(f"{i+1}: Running mutation for: {project.notebook_name}")

        try:
            generate_mutated_accuracy(project)
        except Exception:
            import traceback

            logger.warning(traceback.print_exc())

    with open(internal_path.execution_cache_dir / f"mutation_{k}.finished", "w") as f:
        f.write("finished")


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    n = int(args[0])
    k = int(args[1])

    if len(args) > 2:
        internal_path.training_cache = internal_path.training_cache / args[2]
        internal_path.execution_cache_dir = internal_path.training_cache / "exec_info"
    main(n, k)
