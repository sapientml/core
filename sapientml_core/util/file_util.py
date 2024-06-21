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


import calendar
import datetime
import glob
import json
import os
import time

import pandas as pd


def get_time():
    """Returns the current time.

    Returns
    ----------
    readable : str
        Current time in ISO format
    """
    ts = calendar.timegm(time.gmtime())
    readable = datetime.datetime.fromtimestamp(ts).isoformat()
    return readable


def read_file_in_a_list(file_name):
    """Open a file and place it in a list line by line(read().splitlines()).

    Parameters
    ----------
    file_name : FileDescriptorOrPath
        File name.

    Returns
    ----------
    lines : list[str]
        List file contents line by line.
    """
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return lines


def read_file(file_name):
    """Open file and read data with read().

    Parameters
    ----------
    file_name : FileDescriptorOrPath
        File name.

    Returns
    ----------
    lines : str
        The entire text file read.
    """
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.read()
    return lines


def write_content_to_file(file_name, content):
    """write content to file.

    Parameters
    ----------
    file_name : FileDescriptorOrPath
        File name.
    content : str
        What to write to the file.
    """
    with open(file_name, "w", encoding="utf-8") as out_file:
        out_file.write(content)


def get_file_list(path, type):
    """Get a list of files of a specified type in a directory.

    Parameters
    ----------
    path : FileDescriptorOrPath
        Directory path.
    type : str
        File extension.
    Returns
    ----------
    files_with_given_type : list
        List of retrieved files.
    """
    os.chdir(path)
    files_with_given_type = []
    for file in glob.glob("*." + type):
        files_with_given_type.append((path + "/" + file))
    return files_with_given_type


def load_json(file_name):
    """Load json format file.

    Parameters
    ----------
    file_name : FileDescriptorOrPath
        File name.

    Returns
    ----------
    content : Any
        Loaded content.
    """
    with open(file_name, "r", encoding="utf-8") as input_file:
        content = json.load(input_file)
    return content


def read_csv(csv_path, notebook_path):
    """Read a csv file.

    Parameters
    ----------
    csv_path :  FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str]
        Csv file Path.
    notebook_path : pathlib.Path
        Notebook Directory Path.

    Returns
    ----------
    dataset : pd.DataFrame
        Contents of the loaded csv.
    """

    def read(path, **kwargs):
        if str(path).endswith(".csv"):
            return pd.read_csv(path, encoding_errors="ignore", on_bad_lines="warn", **kwargs)
        return pd.read_table(path, encoding_errors="ignore", on_bad_lines="warn", **kwargs)

    encoding = get_dataset_encoding(notebook_path)
    dataset = read(csv_path, encoding=encoding)
    num_of_features = dataset.shape[1] - 1
    if num_of_features == 0:
        dataset = read(csv_path, encoding=encoding, delim_whitespace=True)
        num_of_features = dataset.shape[1] - 1
    if num_of_features == 0:
        dataset = read(csv_path, encoding=encoding, delimiter=";")
        num_of_features = dataset.shape[1] - 1
    return dataset


def get_dataset_encoding(notebook_path):
    """Get dataset encoding.

    Parameters
    ----------
    notebook_path : StrPath | None | BytesPath
        Directory path of notebooks.

    Returns
    ----------
    encoding : str | None
    """
    if os.path.isdir(notebook_path):
        return None
    if not str(notebook_path).endswith(".py"):
        return None
    encoding = get_dataset_file(notebook_path)
    if encoding:
        return encoding
    return None


def get_dataset_file(notebook_path):
    """Read notebook_path and get encoding.

    Parameters
    ----------
    notebook_path : str
        File name.

    Returns
    ----------
    csv_file_name : str | bytes | None
        File name of csv(dataset).
    encoding : str | None
        Encoding of notebook_path.
    """
    f = open(notebook_path, "r", encoding="utf-8")
    lines = f.readlines()
    f.close()
    encoding = None
    for index in range(len(lines)):
        if ".read_csv(" in lines[index]:
            if "encoding=" in lines[index]:
                encoding = lines[index].split("encoding=")[1].split(")")[0].split(",")[0][1:-1]
            elif "encoding = " in lines[index]:
                encoding = lines[index].split("encoding = ")[1].split(")")[0].split(",")[0][1:-1]
            else:
                encoding = None
            return encoding
    return encoding
