from sklearn.model_selection import TimeSeriesSplit


def split_dataset(dataset, split_column_name, split_num, split_index):
    dataset = dataset.sort_values(split_column_name)
    splitter = TimeSeriesSplit(n_splits=split_num)
    train_idx, test_idx = list(splitter.split(dataset))[split_index]
    train_dataset, test_dataset = dataset.iloc[train_idx], dataset.iloc[test_idx]
    for col in train_dataset.columns:
        if train_dataset[col].isnull().all():
            if test_dataset[col].dtype == float or test_dataset[col].dtype == int:
                train_dataset.loc[:, col] = 0
            elif test_dataset[col].dtype == object:
                train_dataset.loc[:, col] = ""
            elif test_dataset[col].dtype == bool:
                train_dataset.loc[:, col] = False
    return train_dataset, test_dataset
