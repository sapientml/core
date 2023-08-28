import calendar
import datetime
import glob
import json
import os
import time

import pandas as pd


def get_time():
    ts = calendar.timegm(time.gmtime())
    readable = datetime.datetime.fromtimestamp(ts).isoformat()
    return readable


def read_file_in_a_list(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return lines


def read_file(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.read()
    return lines


def write_content_to_file(file_name, content):
    with open(file_name, "w", encoding="utf-8") as out_file:
        out_file.write(content)


def get_file_list(path, type):
    os.chdir(path)
    files_with_given_type = []
    for file in glob.glob("*." + type):
        files_with_given_type.append((path + "/" + file))
    return files_with_given_type


def load_json(file_name):
    with open(file_name, "r", encoding="utf-8") as input_file:
        content = json.load(input_file)
    return content


def read_csv(csv_path, proj_name, notebooks_dir, dataset_dir):
    def read(path, **kwargs):
        if str(path).endswith(".csv"):
            return pd.read_csv(path, **kwargs)
        return pd.read_table(path, **kwargs)

    encoding = get_dataset_encoding(proj_name, notebooks_dir, dataset_dir)
    dataset = read(csv_path, encoding=encoding)
    num_of_features = dataset.shape[1] - 1
    if num_of_features == 0:
        dataset = read(csv_path, encoding=encoding, delim_whitespace=True)
        num_of_features = dataset.shape[1] - 1
    if num_of_features == 0:
        dataset = read(csv_path, encoding=encoding, delimiter=";")
        num_of_features = dataset.shape[1] - 1
    return dataset


def get_dataset_encoding(proj_name, notebooks_dir, dataset_dir):
    notebook_files = os.listdir(notebooks_dir / proj_name)
    for notebook_file in notebook_files:
        notebook_path = notebooks_dir / proj_name / notebook_file
        if os.path.isdir(notebook_path):
            continue
        if not str(notebook_path).endswith(".py"):
            continue
        _, encoding = get_dataset_file(notebook_path, proj_name, dataset_dir)
        if encoding:
            return encoding
    return None


def get_dataset_file(notebook_path, proj_name, dataset_dir):
    f = open(notebook_path, "r", encoding="utf-8")
    lines = f.readlines()
    f.close()
    csv_file_name = os.listdir(dataset_dir / proj_name)[0]  # assume only one csv file
    encoding = None
    for index in range(len(lines)):
        if ".read_csv(" in lines[index]:
            if "encoding=" in lines[index]:
                encoding = lines[index].split("encoding=")[1].split(")")[0].split(",")[0][1:-1]
            elif "encoding = " in lines[index]:
                encoding = lines[index].split("encoding = ")[1].split(")")[0].split(",")[0][1:-1]
            else:
                encoding = None
            return csv_file_name, encoding
    return csv_file_name, encoding
