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

from sapientml_core import internal_path

LABELS_TO_IGNORE_NOW = {
    "PREPROCESS:DeleteColumns:drop:pandas",
    "PREPROCESS:Category:map:pandas",
    "PREPROCESS:MissingValues:dropna:pandas",
    "PREPROCESS:Category:replace:pandas",
    "PREPROCESS:FeatureSelection:select_dtypes:pandas",
    "PREPROCESS:GenerateColumn:addition:pandas",
}


def main():
    """Removes duplication of labelling orders from dependent_labels.json file.

    This scripts create the dataflow model, i.e., extracts the order of two APIs A and B if there is any.
    There is an order between A --> B if A and B are dependent on each other based on 'dependent_api_extractor.py' and
    A is always followed by B in all piplelines and there is NO case in the corpus where B is followed by A.

    """
    with open(internal_path.training_cache / "dependent_labels.json", "r", encoding="utf-8") as dependent_api_file:
        dependent_labels = json.load(dependent_api_file)

    dependent_order = set()

    for dependent_label_str in dependent_labels.keys():
        dep_str_after_bracket_removal = dependent_label_str.replace("[", "").replace("]", "").replace("'", "")
        parts = dep_str_after_bracket_removal.split(",")
        if (parts[0] in LABELS_TO_IGNORE_NOW) or (parts[1].strip() in LABELS_TO_IGNORE_NOW):
            continue
        first = parts[0].split(":")[1].strip()
        second = parts[1].split(":")[1].strip()
        inverse_order = second + "#" + first
        if first != second:
            if inverse_order in dependent_order:
                dependent_order.remove(inverse_order)
            else:
                dependent_order.add(parts[0].strip() + "#" + parts[1].strip())

    with open(internal_path.training_cache / "label_order.json", "w", encoding="utf-8") as outfile:
        json.dump(list(dependent_order), outfile, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="Tag for output files and dirs.")
    args = parser.parse_args()
    if args.tag:
        internal_path.training_cache = internal_path.training_cache / args.tag

    main()
