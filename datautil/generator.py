import numpy as np
import pandas as pd


class FuncGen:
    def __init__(
        self,
        num_dims,
        num_random_features=500,
        alpha=1.0,
        length_scale=0.25,
    ):

        self.N = num_random_features
        self.d = num_dims
        self.w = np.random.randn(self.N, self.d)
        self.b = np.random.uniform(0, 2 * np.pi, self.N)
        self.theta = np.random.randn(self.N)
        self.alpha = alpha
        length_scale = np.random.uniform(0.1, 0.5)
        self.l = length_scale * num_dims  # Smoother functions in higher dimensions

    def evaluate_function(self, x):
        output = np.apply_along_axis(
            lambda x: np.sum(
                self.theta
                * np.sqrt(2.0 * self.alpha / self.N)
                * np.cos(np.dot(self.w, x / self.l) + self.b)
            ),
            1,
            x,
        )
        return output


class GenerateDataset:

    def __init__(self):

        pass

    def _valid_class_prop(self, y, alpha):
        unique, counts = np.unique(y, return_counts=True)
        normalized_counts = counts / counts.sum()
        return len(unique) == 2 and all(normalized_counts >= alpha)

    def _classify_output(self, y):
        # y = np.sign(y)
        threshold = np.median(y)
        y = (y > threshold).astype(int)
        return y

    def _gen_data(
        self,
        num_dims,
        num_tasks,
        num_instances,
        num_clusters,
        cluster_size,
    ):

        common_weight = 0.90
        specific_weight = 1 - common_weight

        if not cluster_size:
            self.tasks_per_cluster = [num_tasks // num_clusters] * num_clusters
            for i in range(num_tasks % num_clusters):
                self.tasks_per_cluster[i] += 1
        else:
            self.tasks_per_cluster = cluster_size
            if len(self.tasks_per_cluster) != num_clusters:
                raise ValueError(
                    f"The size of the cluster should be {num_clusters}, not {len(self.tasks_per_cluster)}."
                )

        while True:
            X, Y = [], []
            success = True

            for cluster_idx in range(num_clusters):

                cluster_funcgen = FuncGen(num_dims=num_dims)

                for _ in range(self.tasks_per_cluster[cluster_idx]):
                    funcgen_specific = FuncGen(num_dims)
                    x = np.random.uniform(size=(num_instances, num_dims)) * 2.0 - 1.0
                    y = (cluster_funcgen.evaluate_function(x) * common_weight) + (
                        specific_weight * funcgen_specific.evaluate_function(x)
                    )
                    if not self.regression:
                        y = self._classify_output(y)
                        if not self._valid_class_prop(y, alpha=0.1):
                            success = False
                            break  # Imbalanced classification, restart entire generation
                    X.append(x)
                    Y.append(y)

                if not success:
                    break  # Restart if any task failed

            # Only return if all clusters and tasks generated successfully
            if success and len(X) == num_tasks:
                return X, Y

    def __call__(
        self,
        regression,
        num_dims,
        num_tasks,
        num_instances,
        num_clusters,
        cluster_size=None,
    ):

        self.regression = regression

        def _gen_df(x, y, task_num):
            columns = [f"Feature {i}" for i in range(x.shape[1])]
            columns.append("Target")
            df = pd.DataFrame(
                np.column_stack((x, y)),
                columns=columns,
            )
            df["Task"] = np.ones_like(y) * task_num
            return df

        x_list, y_list = self._gen_data(
            num_dims,
            num_tasks,
            num_instances,
            num_clusters,
            cluster_size,
        )

        dfs = []
        task_counter = 0
        for cluster_idx in range(num_clusters):
            for _ in range(self.tasks_per_cluster[cluster_idx]):
                x = x_list[task_counter]
                y = y_list[task_counter]
                dfs.append(_gen_df(x, y, task_counter))
                task_counter += 1

        return pd.concat(dfs, ignore_index=True)
