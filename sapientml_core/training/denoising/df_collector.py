# Copyright 2023 The SapientML Authors
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

import pandas as pd


def update_column_names(collector, line_no, obj, obj_name):
    now_obj = obj
    if isinstance(now_obj, pd.Series):
        now_obj = now_obj.to_frame()
    if isinstance(now_obj, pd.DataFrame):
        collector[line_no] = (list(now_obj.columns), obj_name, str(type(now_obj)))
    else:
        collector[line_no] = (None, obj_name, str(type(now_obj)))
    return collector
