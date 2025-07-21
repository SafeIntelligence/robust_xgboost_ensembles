"""
A decision tree node specifically designed to be used 
by trees in the XGBoost ensemble
"""
from typing import Optional

import numpy as np

from .dt_node import DTNode

class XGBoostDTNode(DTNode):
    
    def __init__(self,
                 prediction: float,
                 depth: int,
                 all_data: Optional[np.ndarray] = None,
                 all_annotations: Optional[np.ndarray] = None,
                 all_weights: Optional[np.ndarray] = None,
                 all_previous_predictions: Optional[np.ndarray] = None,
                 training_data_indices: Optional[np.ndarray] = None,
                 idx: int = 0,
                 cover: float = 0):

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
            all_previous_predictions:
                The previous predicitons in the ensemble to compute gradients
            training_data_indices:
                The indices of the data used to train this node of shape
                (n_indices,).
            idx:
                The index of the node wrt to other nodes at the same depth.
            cover:
                The sum of the hessians (second derivatives) at the node
        """
        self._all_previous_predictions = all_previous_predictions
        self._cover = cover
        
        super().__init__(
            prediction = prediction,
            depth = depth,
            all_data = all_data,
            all_annotations = all_annotations,
            all_weights = all_weights,
            training_data_indices = training_data_indices,
            idx = idx
        )
    
    @property
    def parent_node(self):
        return self._parent_node
    
    @parent_node.setter
    def parent_node(self, value):
        self._parent_node = value
    
    @property
    def node_loss_reduction(self):
        return self._node_loss_reduction
    
    @node_loss_reduction.setter
    def node_loss_reduction(self, value):
        self._node_loss_reduction = value
        
    @property
    def num_training_datapoints(self):
        data, _, _ , _ = self.train_set
        return len(data)
        
    @property
    def train_set(self):

        """
        Returns the training data, annotations and weights used to train this
        node specifically, as indicated by self._indices.
        """
        
        weights_in_node = None
        previous_predictions_in_node = None
        
        if self._all_weights is not None:
            weights_in_node = self._all_weights[self._training_data_indices]
            
        if self._all_previous_predictions is not None:
            previous_predictions_in_node = self._all_previous_predictions[self._training_data_indices]

        return (
            self._all_training_data[self._training_data_indices],
            self._all_annotations[self._training_data_indices],
            weights_in_node,
            previous_predictions_in_node
            )
    
    def encode_split(self,
                     split_feature_idx: int,
                     split_value: float,
                     indices_left: np.ndarray,
                     indices_right: np.ndarray,
                     left_child_prediction: float,
                     right_child_prediction: float,
                     left_cover: float,
                     right_cover: float,
                    ):

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
            left_child_prediction:
                The prediction of the left child node
            right_child_prediction:
                The prediction of the right child node
            left_cover:
                The cover (sum of hessians) on the left child
            right_cover:
                The cover of the right child
        """

        self._split_feature_idx = split_feature_idx
        self._split_value = split_value

        # layer_idx is the index of the current node in the layer.
        layer_idx = self._idx - 2 ** self._depth + 1
        left_idx = 2 * layer_idx + 2 ** (self.depth + 1) - 1
        right_idx = 2 * layer_idx + 2 ** (self.depth + 1) + 1 - 1

        self._left_child = XGBoostDTNode(prediction=left_child_prediction,
                                  depth=self._depth + 1,
                                  all_data=self._all_training_data,
                                  all_annotations=self._all_annotations,
                                  all_weights=self._all_weights,
                                  all_previous_predictions=self._all_previous_predictions,
                                  training_data_indices=indices_left,
                                  idx=left_idx,
                                  cover=left_cover)
        
        self._left_child.parent_node = self

        self._right_child = XGBoostDTNode(prediction=right_child_prediction,
                                   depth=self._depth + 1,
                                   all_data=self._all_training_data,
                                   all_annotations=self._all_annotations,
                                   all_weights=self._all_weights,
                                   all_previous_predictions=self._all_previous_predictions,
                                   training_data_indices=indices_right,
                                   idx=right_idx,
                                   cover=right_cover)
        
        self._right_child.parent_node = self
        
    def predict(self, data: np.array) -> float:

        """
        Predicts the output for the given data. Note that this is
        overriden in this class as XGBoost assigns points to the 
        left child node if the feature value is *less* than the 
        threshold. (The other implementation in this package assigns
        points to the left child node if the feature value is *less
        than or equal to* the threshold.)

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
        
        
    def all_possible_predictions(self, data: np.ndarray, pert_radii) -> float:

        """
        Predicts the worst case output for the given data.

        Args:
            data:
                The input data of shape (n_datapoints,n_features).
        """
        
        predictions=set()
        
        if self._split_feature_idx is None:
            predictions.add(self._prediction)
            return predictions
        
        feature_value = data[self._split_feature_idx]

        # Calculate the perturbed range
        feature_lower = feature_value - pert_radii[self._split_feature_idx]
        feature_upper = feature_value + pert_radii[self._split_feature_idx]
            
        if feature_lower <= self._split_value:
            predictions.update(
                self._left_child.all_possible_predictions(data, pert_radii)
            )

        # Check overlaps with the "no" branch (> split_condition)
        if feature_upper > self._split_value:
            predictions.update(
                self._right_child.all_possible_predictions(data, pert_radii)
            )
            
        return predictions
        
        