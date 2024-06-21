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

import datetime


class Code_Template:
    """Code Template class."""

    def __init__(self):
        self.str_reverse = {"NOW": str(datetime.datetime.now())}

    def update(self, lines):
        """update method.

        Parameters
        ----------
        lines : list[str]
            A line in block code from jupyter content template.

        Returns
        -------
        out : list[str]
            Updated line in block code from jupyter content template.

        """
        out = []
        for line in lines:
            for key in self.str_reverse:
                line = line.replace(key, self.str_reverse[key])
            out.append(line)
        return out
