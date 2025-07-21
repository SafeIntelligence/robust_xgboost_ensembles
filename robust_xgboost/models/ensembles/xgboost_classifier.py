"""
XGBoost Binary Classification Implementation using Taylor-Series 
Approximation of Loss and Regularisation

Note: The classifier currently only supports binary classification
"""

import json

import numpy as np
from tqdm import tqdm

from typing import Union

from ..decision_trees.losses import XGBoostLogisticLoss
from .xgboost import XGBoostBase

from ..decision_trees.xgboost_tree import XGBoostTree


class XGBoostClassifier(XGBoostBase):
    
    def __init__(
        self,
        n_base_models: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.2,
        lamda: float = 0.0,
        gamma: float = 0.0,
        initial_pred: float = 0.5,
        min_samples_leaf: int = 1,
        pert_radius: Union[float, np.ndarray] = 0,
        rob_alpha: float = 1.0,
        scale_pos_weight: float = 1.0,
    ):

        """
        Args:
            n_base_models:
                The number of boosting stages to perform.
            learning_rate:
                Weight applied to each classifier at each boosting iteration.
            max_depth: 
                The maximum permitted depth of individual decision trees
            lamda:
                The L2 regularisation parameter
            gamma:
                The minimum loss decrease required for splitting
            initial_pred:
                The initial probability prediction of the classifier
            min_samples_leaf:
                The minimum number of datapoints that a leaf is permitted to have
            pert_radius:
                Either a float or numpy array indicating the perturbation
                radius for robust training. Set to 0 for standard training.
            rob_alpha:
                The weightage of the robust loss in the loss function. A value
                of 1.0 indicates that the robust loss is used exclusively.
            scale_pos_weight:
                The scaling factor for the positive class in the loss function.
        """
        
        super().__init__(
            loss = XGBoostLogisticLoss(),
            n_base_models = n_base_models,
            max_depth = max_depth,
            learning_rate = learning_rate,
            lamda = lamda,
            gamma = gamma,
            initial_pred = initial_pred,
            min_samples_leaf = min_samples_leaf,
            pert_radius = pert_radius,
            rob_alpha = rob_alpha
        )
        
        self.scale_pos_weight = scale_pos_weight
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid: clamps x to avoid overflow in exp.
        """
        x_clamped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clamped))

    def fit(
        self, 
        data: np.ndarray,
        annotations: np.ndarray,
        weights: np.ndarray = None
    ):
        """
        Fits the model to the given training set.

        Args:
            data:
                The input samples of shape (n_samples, n_features).
            annotations:
                The data annotations of shape (n_samples,).
            weights:
                The sample weights of shape (n_samples,).

        Returns:
            The fitted model.
        """
        
        if self.scale_pos_weight != 1.0 and weights is None:
            weights = np.array([1.0 if y == 0 else self.scale_pos_weight for y in annotations])
    
        if weights is not None:
            
            weights = weights * len(weights)/sum(weights)
            
        predicted_prob = np.array([self.initial_pred] * len(annotations))
            
        predicted_logits = self.loss.convert_probability_to_logits(predicted_prob)
        
        if len(data.shape) != 2 or len(annotations.shape) != 1 or \
                (weights is not None and len(weights.shape) != 1):
            raise ValueError("Expected x to be of shape (n_samples, n_features), "
                             "and y and weights to be of shape (n_samples,)")

        if data.shape[0] != annotations.shape[0] or \
                (weights is not None and data.shape[0] != weights.shape[0]):
            raise ValueError("Expected same number of samples in x, y and weights")

        if data.shape[0] <= 1:
            raise ValueError("Expected input data to have more than 1 sample")

        for iboost in tqdm(range(self.num_base_models)):
            
            base_model = self._base_model.clone()
            
            base_model.fit( data,
                            annotations,
                            weights,
                            predicted_prob)
                
            predicted_logits = predicted_logits + base_model.predict(data)

            # use stable sigmoid here
            try:
                predicted_prob = self._sigmoid(predicted_logits)
            except FloatingPointError:
                print(f"numerical instability in iteration {iboost}")
                raise

            if base_model._root_node.is_leaf:
                break
            
            self._base_models_.append(base_model)
            
    def predict_proba(self, data: np.ndarray):
        """
        Predicts probabilities for X. 
        Currently only binary classification supported.

        Args:
            data:
                array of shape (n_samples, n_features). The input samples.
        Returns:
            The predicted probabilities, array of shape (n_samples, 1).
        """
        
        initial_probability = np.array([self.initial_pred]*len(data))
        logits = self.loss.convert_probability_to_logits(initial_probability)
        
        for base_model in self._base_models_:
            logits += self.learning_rate * base_model.predict(data)

        # use stable sigmoid here
        probability = self._sigmoid(logits)
        return probability
            
    def predict(self, data: np.ndarray, threshold: float = 0.5):

        """
        Predicts classes for X. 
        Currently only binary classification supported.

        Args:
            data:
                array of shape (n_samples, n_features). The input samples.
            threshold:
                Float, that describes the probability threshold for 
                classification
        Returns:
            The predicted values, array of shape (n_samples,).
        """
        
        probability = self.predict_proba(data)
            
        output = probability > threshold
            
        output = output.astype(int)
        
        return output
