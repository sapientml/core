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


from pathlib import Path

sapientml_core_root = Path(__file__).parents[0]

adaptation_root_dir = sapientml_core_root / "adaptation"
artifacts_path = adaptation_root_dir / "artifacts"
model_path = sapientml_core_root / "models"

benchmark_path = sapientml_core_root / "benchmarks"
corpus_path = sapientml_core_root / "corpus"
training_cache = sapientml_core_root / ".cache"

execution_cache_dir = training_cache / "exec_info"
analysis_dir = training_cache / "analysis"
clean_notebooks_dir_name = "clean-notebooks"
clean_dir = corpus_path / clean_notebooks_dir_name
project_labels_path = corpus_path / "annotated-notebooks" / "annotated-notebooks-1140.csv"
