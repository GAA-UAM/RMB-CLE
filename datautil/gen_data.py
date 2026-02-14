#%%
import os
import numpy as np
import pandas as pd
from generator import GenerateDataset
from sklearn.model_selection import train_test_split


def gen_data(
    regression,
    num_dims,
    num_tasks,
    num_instances,
    num_clusters,
    num_train_per_task,
    num_test_per_task,
    batch,
    cluster_size=None,
):

    gen_data = GenerateDataset()
    df = gen_data(
        regression=regression,
        num_dims=num_dims,
        num_tasks=num_tasks,
        num_instances=num_instances,
        num_clusters=num_clusters,
        cluster_size=cluster_size,
    )

    train_dfs, test_dfs = [], []
    for i, task_id in enumerate(df["Task"].unique()):
        task_df = df[df["Task"] == task_id]

        train_task_df, test_task_df = train_test_split(
            task_df,
            train_size=num_train_per_task,
            test_size=num_test_per_task,
        )

        train_dfs.append(train_task_df)
        test_dfs.append(test_task_df)

        combined_train_df = pd.concat(train_dfs)
        combined_test_df = pd.concat(test_dfs)

    path = os.path.join(os.path.dirname(__file__), "..", "toy_datasets")
    os.chdir(path)
    os.makedirs(f"{num_clusters}clusters", exist_ok=True)
    os.chdir(f"{num_clusters}clusters")
    os.makedirs(f"batch{batch}", exist_ok=True)
    os.chdir(f"batch{batch}")

    dataset_type = "reg" if regression else "clf"
    combined_train_df.to_csv(f"train_{dataset_type}.csv", index=False)
    combined_test_df.to_csv(f"test_{dataset_type}.csv", index=False)


if __name__ == "__main__":
    # for batch in range(1, 101):
    for batch in range(1, 2):
        np.random.seed(batch)
        print(f"Generating data for batch {batch}")
        # for clf in [True, False]:
        for clf in [False]:
            gen_data(
                regression=clf,
                num_dims=5,
                num_tasks=25,
                num_instances=300,
                num_clusters=5,
                num_train_per_task=100,
                num_test_per_task=200,
                batch=batch,
                cluster_size=[3, 4, 5, 8, 10],
            )
