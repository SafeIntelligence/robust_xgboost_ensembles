"""
The Decision Tree class.
"""

import json
from typing import Optional, Union, Callable, List, Tuple

import numpy as np

from .dt_node import DTNode


class DecisionTree:

    """
    A class implementing standard Decision Trees.
    """

    def __init__(self,
                 loss_func: Optional[Callable],
                 pert_radius: Union[float, list, np.ndarray], max_depth: int,
                 min_samples_leaf: int):

        """
        Args:
            loss_func:
                The loss functions for choosing the best split.
            pert_radius:
                Either a float or numpy array indicating the perturbation
                radius for robust training. Set to 0 for standard training.
            max_depth:
                The maximum depth of the tree.
            min_samples_leaf:
                The minimum number of datapoints required to split an internal
                node.
        """

        self._pert_radius = float(pert_radius) if isinstance(pert_radius, int) else pert_radius

        if min_samples_leaf < 1:
            raise ValueError("min_datapoints must be >= 1")
        if max_depth is None or max_depth < 1:
            raise ValueError("max_depth must be set to integer >= 1")

        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._loss_func = loss_func

        self._root_node: Optional[DTNode] = None

    @property
    def num_nodes(self) -> int:
        return len(self._root_node.get_nodes())

    @property
    def num_leaf_nodes(self) -> int:
        return len(self._root_node.get_leaf_nodes())

    @property
    def leaf_nodes(self) -> List[DTNode]:
        return self._root_node.get_leaf_nodes()

    @property
    def loss_func(self) -> Optional[Callable]:
        return self._loss_func

    @property
    def pert_radius(self) -> Union[float, list, np.ndarray]:
        return self._pert_radius

    def reset_fit(self) -> None:

        """
        Resets all parameters set by the fit method.
        """

        self._root_node = None

    def _init_root_node(self,
                        data: np.ndarray,
                        annotations: np.ndarray,
                        weights: Optional[np.ndarray] = None,
                        previous_predictions: Optional[Union[float, np.ndarray]] = None) -> None:

        """
        Initialises the root node.

        Args:
            data:
                The input data of shape (n_datapoints, n_features).
            annotations:
                The targets for the input data.
            weights:
                The weights for the input data.
        """

        raise NotImplementedError()

    def predict(self, data: np.ndarray) -> np.ndarray:

        """
        Predicts the output for the given data.

        Args:
            data:
                The input data of shape (n_datapoints, n_features).

        Returns:
            The predictions for the input data.
        """

        if self._root_node is None:
            raise RuntimeError("Model has not been fitted yet")

        if len(data.shape) != 2:
            raise ValueError("Expected input data to be of shape "
                             "(n_datapoints, n_features)")

        if len(data.shape) == 0:
            raise ValueError("Expected at least one datapoint in data")

        n_datapoints = data.shape[0]
        res = np.zeros(n_datapoints)

        for i in range(n_datapoints):
            res[i] = self._root_node.predict(data[i])

        return res

    def fit(self,
            data: np.ndarray,
            annotations: np.ndarray,
            weights: Optional[np.ndarray] = None,
            previous_predictions: Optional[Union[float, np.ndarray]] = None,
            ) -> None:
        """
        Fits the model to the given training set.

        Args:
            data:
                The input samples of shape (n_samples, n_features).
            annotations:
                The data annotations of shape (n_samples,).
            weights:
                The sample weights of shape (n_samples,).
            previous_predictions:
                The previous predictions in the ensemble (for XGBoost)

        Returns:
            The fitted model.
        """

        self.reset_fit()

        if len(data.shape) != 2 or len(annotations.shape) != 1 or \
                (weights is not None and len(weights.shape) != 1):
            raise ValueError("Expected data to be of shape (n_datapoints, "
                             "n_features), and annotations and weights to be "
                             "of shape (n_datapoints,)")

        if data.shape[0] != annotations.shape[0] or \
                (weights is not None and data.shape[0] != weights.shape[0]):
            raise ValueError("Expected same number of datapoints in data, "
                             "annotations and weights")

        if data.shape[0] <= 1:
            raise ValueError("Expected input data to have more than 1 "
                             "datapoint")
            
        if previous_predictions is None:
            self._init_root_node(data, annotations, weights)
            
        else:
            self._init_root_node(data, annotations, weights, previous_predictions)
            
        node_queue = [self._root_node]

        while len(node_queue) > 0:

            node = node_queue.pop()

            if self._has_reached_stop_condition(node):
                continue

            left_child, right_child = self._fit_one_node(node)

            node_queue.append(left_child)
            node_queue.append(right_child)

    def _fit_one_node(self, node: DTNode) -> Optional[Tuple[DTNode, DTNode]]:

        """
        Fits the best split by brute force looping through all possible splits.

        Args:
            node:
                The node to fit the best split for.

        Returns:
            The left and right child nodes.
        """

        raise NotImplementedError()

    def _has_reached_stop_condition(self, node: Optional[DTNode]) -> bool:

        """
        Checks if the node is None or has reached a stop condition.

        Args:
            node:
                The node to check.

        Returns:
            True if the node has reached a stop condition, otherwise False.
        """

        if node is None:
            return True

        if self._max_depth is not None and node.depth >= self._max_depth:
            return True

        if node.num_training_datapoints < self._min_samples_leaf:
            return True

        return False

    def clone(self) -> 'DecisionTree':

        """
        Returns a decision tree with same hyperparameters but without any
        fitted data.
        """

        raise NotImplementedError()

    def save_model(self, filepath: str) -> None:

        """
        Saves the model in JSON format.

        The format follows the JSON format as used in the XGBoost library.
        However, values that are not used by Venus may be missing.

        Args:
            filepath:
                The path of the savefile
        """

        this_tree_dict = [self._root_node.to_dict()]

        with open(filepath, "w", buffering=1) as f:
            json.dump(this_tree_dict, f, indent=2)

    def load_model(self, filepath: str) -> None:

        """
        Loads the model from JSON format. The format is expected to be as
        produced by save_model.

        Args:
            filepath:
                The path of the savefile
        """

        with open(filepath, "r") as f:
            this_tree_dict = json.load(f)[0]

        self._root_node = DTNode.from_dict(this_tree_dict)