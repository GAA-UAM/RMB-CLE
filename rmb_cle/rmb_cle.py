"""
RMB_CLE: Robust Multi-task Boosting using Clustering and Local Ensembling

A multi-task learning framework that:
1) Learns task similarity by training a simple "residual" model per task.
2) Computes cross-task generalization errors to build a task similarity matrix.
3) Clusters tasks using hierarchical clustering (linkage + fcluster).
4) Trains one model per cluster and uses it for prediction.

License: LGPL-2.1 license

====================================================
Date released : 2025-07-21
Version       : 0.0.2
Update        : 2025-09-06
====================================================
"""

import copy
import numpy as np
from collections import defaultdict
from sklearn.base import BaseEstimator
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, fcluster
from libs._logging import FileHandler, ConsoleHandler
from sklearn.metrics import mean_squared_error, silhouette_score, accuracy_score


def _split_task(X):
    """
    Split input matrix into:
      - X_data: feature matrix (all columns except the last)
      - X_task: task id vector (last column)
    """
    unique_values = np.unique(X[:, -1])
    mapping = {value: index for index, value in enumerate(unique_values)}

    # Replace raw task ids by dense indices 0..T-1
    X[:, -1] = np.vectorize(mapping.get)(X[:, -1])

    X_task = X[:, -1]
    X_data = np.delete(X, -1, axis=1).astype(float)
    return X_data, X_task


class RMB_CLE(BaseEstimator):
    """

    Parameters
    ----------------------------
    residual_model_cls:
        A simple learner used to estimate cross-task similarity.
        (e.g., shallow tree or linear model)
    task_model_cls:
        The multi-task/cluster model class trained on pooled tasks within a cluster.
    residual_model_as_cls:
        If True, the cluster model is the same class as residual_model_cls.
        If False, the cluster model uses task_model_cls.
    n_clusters:
        Number of clusters to cut from hierarchical clustering; "auto" selects via silhouette.
    regression:
        If True uses MSE; if False uses 1-accuracy as error metric.
    """

    def __init__(
        self,
        residual_model_cls,
        task_model_cls,
        residual_model_as_cls,
        n_iter_1st,
        n_iter_3rd,
        max_iter,
        learning_rate,
        regression,
        n_clusters=None,
        random_state=111,
        task_to_cluster_input=None,
    ):
        # Model classes / training controls
        self.residual_model_cls = residual_model_cls
        self.task_model_cls = task_model_cls
        self.residual_model_as_cls = residual_model_as_cls

        # Iteration controls (used by task_model_cls / boosting-style learners)
        self.n_iter_1st = n_iter_1st
        self.n_iter_3rd = n_iter_3rd

        # Shared training controls
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.regression = regression

        # Clustering controls
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.task_to_cluster_input = task_to_cluster_input

        # Learned artifacts (populated after fit)
        self.cluster_models_ = {}  # cluster_id -> trained model
        self.task_to_cluster_ = {}  # task_id -> cluster_id
        self.outlier_tasks_ = set()  # reserved for future use
        self.residual_matrix_ = []  # similarity matrix between tasks
        self.distance_matrix_ = []  # distance matrix between tasks
        self.linkage_matrix_ = []  # hierarchical clustering linkage matrix

        # Loggers
        self.fh_logger = FileHandler()
        self.ch_logger = ConsoleHandler()

        # Reproducibility
        np.random.seed(self.random_state)

        # Hierarchical clustering configuration
        self._linkage_method = "average"
        self._original_linkage = linkage

    def set_linkage_method(self, method_name):
        """Change the hierarchical clustering linkage method (e.g., 'average', 'complete', 'single')."""
        self._linkage_method = method_name

    def construct_task_clusters(self, X, y, task_ids):
        """
        Build task clusters based on a task-to-task similarity matrix.
        """
        unique_tasks = np.unique(task_ids)
        T = len(unique_tasks)

        # Handle degenerate cases early
        if T == 0:
            self.task_to_cluster_ = {}
            return {1: []}
        if T == 1:
            self.task_to_cluster_ = {unique_tasks[0]: 1}
            return {1: [unique_tasks[0]]}

        task_data = {}
        for task in unique_tasks:
            idx = task_ids == task
            task_data[task] = (X[idx], y[idx])

        task_models = {}
        for task in unique_tasks:
            X_task, y_task = task_data[task]
            params = {
                "random_state": self.random_state,
                "max_iter": self.max_iter,
                "learning_rate": self.learning_rate,
                "max_depth": 1,
            }
            model = self.residual_model_cls(**params)
            model.fit(X_task, y_task)
            task_models[task] = copy.deepcopy(model)

        cross_errors = np.zeros((T, T), dtype=float)

        for i, ti in enumerate(unique_tasks):
            Xi, yi = task_data[ti]

            for j, tj in enumerate(unique_tasks):
                mdl = task_models[tj]
                yhat = mdl.predict(Xi)

                if self.regression:
                    # Regression: lower MSE means "more similar"
                    err = mean_squared_error(yi, yhat)
                else:
                    # Classification: use 1 - accuracy as an "error"
                    # If model provides probabilities, convert to class predictions.
                    if hasattr(mdl, "predict_proba"):
                        proba = mdl.predict_proba(Xi)
                        if proba.ndim == 2 and proba.shape[1] == 2:
                            pred = (proba[:, 1] >= 0.5).astype(yi.dtype)
                        else:
                            pred = np.argmax(proba, axis=1)
                    else:
                        pred = yhat

                    acc = accuracy_score(yi, pred)
                    err = 1.0 - acc

                cross_errors[i, j] = err

        eps = 1e-8
        sim_matrix = 1.0 / (cross_errors + eps)
        self.residual_matrix_ = sim_matrix

        distance_matrix = cosine_distances(sim_matrix)

        # Numerical safety: enforce symmetry
        distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)
        self.distance_matrix_ = distance_matrix

        condensed = squareform(distance_matrix)
        self.linkage_matrix_ = self._original_linkage(
            condensed,
            method=self._linkage_method,
        )

        def _pick_k_auto(distance_matrix, linkage_matrix, T, k_cap=20):
            if T <= 2:
                return min(T, 2)

            Kmax = min(T, k_cap)
            best_k, best_score = 2, -1.0

            for k_try in range(2, Kmax + 1):
                labels_try = fcluster(linkage_matrix, k_try, criterion="maxclust")
                try:
                    score = silhouette_score(
                        distance_matrix, labels_try, metric="precomputed"
                    )
                except Exception:
                    score = -1.0

                if score > best_score:
                    best_score, best_k = score, k_try

            return best_k

        if self.n_clusters in (None, "auto"):
            k = _pick_k_auto(self.distance_matrix_, self.linkage_matrix_, T, 20)
        else:
            k = int(self.n_clusters)

        k = max(1, min(k, T))
        cluster_labels = fcluster(self.linkage_matrix_, k, criterion="maxclust")

        clusters = defaultdict(list)
        self.task_to_cluster_ = {}

        for task, label in zip(unique_tasks, cluster_labels):
            label = int(label)
            clusters[label].append(task)
            self.task_to_cluster_[task] = label

        return clusters

    def fit(self, X, y):
        """
        Fit the RMB_CLE pipeline
        """
        X, task_ids = _split_task(X)

        # If user didn't provide a mapping, infer clusters from data
        if not self.task_to_cluster_input:
            clusters = self.construct_task_clusters(X, y, task_ids)
        else:
            self.ch_logger.info("[MTCoLE] Using provided task_to_cluster mapping.")
            clusters = defaultdict(list)
            for task, cluster_id in self.task_to_cluster_input.items():
                clusters[cluster_id].append(task)
                self.task_to_cluster_[task] = cluster_id

        X_full = np.column_stack((X, task_ids))

        for cluster_id, tasks in clusters.items():
            mask = np.isin(task_ids, tasks)

            # Parameters for a multi-task boosting-style model (task_model_cls)
            params_mtgb = {
                "random_state": self.random_state,
                "n_iter_1st": self.n_iter_1st,
                "n_iter_2nd": 0,
                "n_iter_3rd": self.n_iter_3rd,
                "max_depth": 1,
                "subsample": 1.0,
                "learning_rate": self.learning_rate,
            }

            # Parameters for residual_model_cls if used as the cluster model
            params_residual = {
                "random_state": self.random_state,
                "max_iter": self.max_iter,
                "learning_rate": self.learning_rate,
                "early_stopping": False,
            }

            # Choose which class to instantiate for the cluster model
            model = (
                self.task_model_cls(**params_mtgb)
                if not self.residual_model_as_cls
                else self.residual_model_cls(**params_residual)
            )
            model.fit(X_full[mask], y[mask])
            self.cluster_models_[cluster_id] = model
        return self

    def predict(self, X):
        """
        Predict using the cluster model corresponding to each task.
        """
        X_input, task_ids = _split_task(X)
        y_pred = np.empty(X.shape[0])

        X_input = X

        for task in np.unique(task_ids):
            idx = task_ids == task

            cluster_id = self.task_to_cluster_.get(task)
            if cluster_id is None:
                raise ValueError(f"No trained model for task {task}.")
            mdl = self.cluster_models_[cluster_id]
            y_pred[idx] = mdl.predict(X_input[idx])

        return y_pred

    def get_params(self, deep=True):
        """scikit-learn compatibility: return constructor params."""
        return {
            "residual_model_cls": self.residual_model_cls,
            "task_model_cls": self.task_model_cls,
            "residual_model_as_cls": self.residual_model_as_cls,
            "n_iter_1st": self.n_iter_1st,
            "n_iter_3rd": self.n_iter_3rd,
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "regression": self.regression,
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """scikit-learn compatibility: set attributes from kwargs."""
        for k, v in params.items():
            setattr(self, k, v)
        return self
