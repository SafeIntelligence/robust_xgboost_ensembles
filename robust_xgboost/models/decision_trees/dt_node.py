"""
The Decision Tree Node.
"""

from typing import Optional, Callable, List, Tuple, Union

import numpy as np

class DTNode:

    def __init__(self,
                 prediction: float,
                 depth: int,
                 all_data: Optional[np.ndarray] = None,
                 all_annotations: Optional[np.ndarray] = None,
                 all_weights: Optional[np.ndarray] = None,
                 training_data_indices: Optional[np.ndarray] = None,
                 idx: int = 0):

        """
        Args:
            prediction:
                The prediction of this node.
            depth:
                The depth of this node.
            all_data:
                All data used at root node of shape (n_datapoints, n_features).
            all_annotations:
                All annotations corresponding to the
                input data used at root node, of shape (n_datapoints, ).
            all_weights:
                All weights used at the root node of shape (n_datapoints,).
            training_data_indices:
                The indices of the data used to train this node of shape
                (n_indices,).
            idx:
                The index of the node wrt to other nodes at the same depth.
        """

        self._prediction = prediction
        self._depth = depth

        # The following variables are used for training.
        self._all_training_data = all_data
        self._all_annotations = all_annotations
        self._all_weights = all_weights
        self._training_data_indices = training_data_indices
        self._idx = idx

        if all_annotations is not None and training_data_indices is not None:
            mask = all_annotations[training_data_indices] != prediction
            self._misclassified_mask = mask
        elif all_annotations is not None and training_data_indices is None:
            self._misclassified_mask = all_annotations != prediction
        else:
            self._misclassified_mask = None

        # The following variables are used are splitting specific.
        self._split_value: Optional[float] = None
        self._split_feature_idx: Optional[int] = None
        self._left_child: Optional[DTNode] = None
        self._right_child: Optional[DTNode] = None

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def data_indices(self) -> Optional[np.ndarray]:
        return self._training_data_indices

    @property
    def prediction(self) -> float:
        return self._prediction

    @prediction.setter
    def prediction(self, val: float) -> None:
        self._prediction = val

    @property
    def depth(self) -> int:
        return int(self._depth)

    @property
    def split_feature_idx(self) -> int:
        return int(self._split_feature_idx)

    @property
    def split_value(self) -> float:
        return float(self._split_value)

    @property
    def children(self) -> Tuple[Optional['DTNode'], Optional['DTNode']]:

        """
        Returns the children of the node.

        Returns:
            (left_child, right_child)
        """

        return self._left_child, self._right_child

    @property
    def num_training_datapoints(self) -> int:
        data, _, _ = self.train_set
        return len(data)

    @property
    def train_set(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:

        """
        Returns the training data, annotations and weights used to train this
        node specifically, as indicated by self._indices.
        """

        if self._all_weights is None:
            return (self._all_training_data[self._training_data_indices],
                    self._all_annotations[self._training_data_indices],
                    None)
        else:
            return (self._all_training_data[self._training_data_indices],
                    self._all_annotations[self._training_data_indices],
                    self._all_weights[self._training_data_indices])

    @property
    def full_train_set(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:

        """
        Returns the full training data, annotations and weights stored in this
        node.
        """

        return self._all_training_data, self._all_annotations, self._all_weights

    @property
    def misclassified_indices(self) -> np.ndarray:

        """
        The misclassified indices of this node.
        """

        return self._training_data_indices[self._misclassified_mask]

    @property
    def is_leaf(self) -> bool:

        return (self._left_child is None) and (self._right_child is None)

    def get_nodes(self) -> List['DTNode']:

        """
        Returns all nodes of the tree.

        Returns:
            A list of the leaf nodes.
        """

        if self._left_child is None and self._right_child is None:
            return [self]

        elif self._left_child is not None and self._right_child is not None:
            return ([self] +
                    self._left_child.get_nodes() +
                    self._right_child.get_nodes())
        else:
            raise RuntimeError(
                "Unexpectedly encountered a node with only one child")

    def get_leaf_nodes(self) -> List['DTNode']:

        """
        Returns the leaf nodes of the tree.

        Returns:
            A list of the leaf nodes.
        """

        if self._left_child is None and self._right_child is None:
            return [self]

        elif self._left_child is not None and self._right_child is not None:
            return self._left_child.get_leaf_nodes() + \
                self._right_child.get_leaf_nodes()
        else:
            raise RuntimeError(
                "Unexpectedly encountered a node with only one child")

    def get_leaf_misclassified(self) -> np.ndarray:

        """
        Returns a tensor with the number of leaf nodes in which each index is
        misclassified.

        Returns:
            A list of the misclassified indices.
        """

        leaf_nodes = self.get_leaf_nodes()
        misclassified_count = np.zeros(len(self._all_training_data))

        for node in leaf_nodes:
            misclassified_count[node.misclassified_indices] += 1

        return misclassified_count

    def predict(self, data: np.ndarray) -> float:

        """
        Predicts the output for the given data.

        Args:
            data:
                The input data of shape (n_datapoints,n_features).
        """

        if self._split_feature_idx is None:
            return self._prediction

        elif data[self._split_feature_idx] < self._split_value:
            return self._left_child.predict(data)

        else:
            return self._right_child.predict(data)

    def encode_split(self,
                     split_feature_idx: int,
                     split_value: float,
                     indices_left: np.ndarray,
                     indices_right: np.ndarray,
                     node_prediction: Callable[[np.ndarray, Optional[np.ndarray]], float]) -> None:

        """
        Encodes the split into the class variables.

        Args:
            split_feature_idx:
                The feature to split on.
            split_value:
                The value to split on.
            indices_left:
                The indices of the datapoints corresponding to the left split.
            indices_right:
                The indices of the datapoints corresponding to the right split.
            node_prediction:
                A function that given the annotations and weights predicts the
                output of the node. For classification trees, this method
                typically predicts the weighted majority class.
        """

        annotations_l = self._all_annotations[indices_left]
        annotations_r = self._all_annotations[indices_right]

        weights_left = self._all_weights[indices_left] if \
            self._all_weights is not None else None
        weights_right = self._all_weights[indices_right] if \
            self._all_weights is not None else None

        predicted_class_left = node_prediction(annotations_l, weights_left)
        predicted_class_right = node_prediction(annotations_r, weights_right)
        self._split_feature_idx = split_feature_idx
        self._split_value = split_value

        # layer_idx is the index of the current node in the layer.
        layer_idx = self._idx - 2 ** self._depth + 1
        left_idx = 2 * layer_idx + 2 ** (self.depth + 1) - 1
        right_idx = 2 * layer_idx + 2 ** (self.depth + 1) + 1 - 1

        self._left_child = DTNode(prediction=predicted_class_left,
                                  depth=self._depth + 1,
                                  all_data=self._all_training_data,
                                  all_annotations=self._all_annotations,
                                  all_weights=self._all_weights,
                                  training_data_indices=indices_left,
                                  idx=left_idx)

        self._right_child = DTNode(prediction=predicted_class_right,
                                   depth=self._depth + 1,
                                   all_data=self._all_training_data,
                                   all_annotations=self._all_annotations,
                                   all_weights=self._all_weights,
                                   training_data_indices=indices_right,
                                   idx=right_idx)

    def to_dict(self) -> dict:

        """
        Returns a dictionary representation of the node and its children.

        Returns:
            A dictionary representation of the node and its children.
        """

        if not self.is_leaf:
            node_dict = {"nodeid": self.idx,
                         "depth": self.depth,
                         "split": f"f{self.split_feature_idx}",
                         "split_condition": self.split_value,
                         "yes": self._left_child.idx,
                         "no": self._right_child.idx,
                         "missing": self._left_child.idx,
                         "children": [self._left_child.to_dict(),
                                      self._right_child.to_dict()]}
        else:
            node_dict = {"nodeid": self.idx,
                         "depth": self.depth,
                         "leaf": self._prediction}

        return node_dict

    # noinspection PyTypeChecker
    @staticmethod
    def from_dict(node_dict: dict) -> 'DTNode':

        """
        Initialises the node and dict with from a dictionary representation.

        The representation is expected to be as produced by the to_dict method.

        Args:
            node_dict:
                The dictionary representation of the node.
        """

        if "children" in node_dict.keys():

            node = DTNode(prediction=-1, depth=node_dict["depth"])
            node._idx = node_dict["nodeid"]
            node._split_feature_idx = int(node_dict["split"][1:])
            node._split_value = node_dict["split_condition"]

            if not node_dict["children"][0]["nodeid"] == node_dict["yes"]:
                raise RuntimeError(
                    "Unexpected missmatch between node_dict['yes'] and "
                    "left child")
            if not node_dict["children"][1]["nodeid"] == node_dict["no"]:
                raise RuntimeError(
                    "Unexpected missmatch between node_dict['no'] and "
                    "right child")

            node._left_child = DTNode.from_dict(node_dict["children"][0])
            node._right_child = DTNode.from_dict(node_dict["children"][1])

        else:
            node = DTNode(prediction=node_dict["leaf"],
                          depth=node_dict["depth"])
            node._idx = node_dict["nodeid"]

        return node
