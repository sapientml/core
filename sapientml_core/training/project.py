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

from dataclasses import dataclass


@dataclass
class ProjectInfo:
    pipeline_path: str  # full path
    dataset_path: str  # full path
    file_name: str  # only name of the pipeline
    notebook_name: str  # only name of the pipeline without extension
    accuracy: float
    csv_name: str
    target_column_name: str
    metric: str
