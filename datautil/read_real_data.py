import os
import pandas as pd
from sklearn.model_selection import train_test_split


def _preprocess_adult_data(df, task_column):

    unique_values = sorted(df[task_column].unique())
    value_map = {value: index for index, value in enumerate(unique_values)}
    df["Task"] = df[task_column].map(value_map)

    columns_to_drop = ["sex", "race"]
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    if existing_columns_to_drop:
        df.drop(columns=existing_columns_to_drop, inplace=True)

    return df


def ReadData(dataset, random_state):

    special_datasets = {"adult_gender": "sex", "adult_race": "race"}

    base_dataset = "adult" if dataset in special_datasets else dataset
    task_col = special_datasets.get(dataset)

    base_path = os.path.join(os.path.dirname(__file__), "..", "real_case_datasets", base_dataset)

    def load_csv(name):
        return pd.read_csv(os.path.join(base_path, f"{name}.csv"))

    if dataset in [
        "adult_gender",
        "adult_race",
    ]:
        data = load_csv(f"{base_dataset}_data")
        target = load_csv(f"{base_dataset}_target")
        data = _preprocess_adult_data(data, task_col)
    else:
        data = load_csv(f"{dataset}_data")
        target = load_csv(f"{dataset}_target")
    data_train, data_test, target_train, target_test = train_test_split(
        data, target, test_size=0.2, random_state=random_state
    )

    if target_train.ndim > 1:
        target_train = target_train.values.ravel()
        target_test = target_test.values.ravel()
    else:
        target_train = target_train.values
        target_test = target_test.values

    return (
        data_train.values,
        target_train,
        data_test.values,
        target_test,
    )
