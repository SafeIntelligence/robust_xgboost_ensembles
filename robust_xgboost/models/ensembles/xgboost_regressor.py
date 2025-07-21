"""
XGBoost Regression Implementation using 
Taylor-Series Approximation of Loss and Regularisation
"""

import json

import numpy as np
from tqdm import tqdm

from typing import Union

from ..decision_trees.losses import XGBoostMSELoss
from .xgboost import XGBoostBase

from ..decision_trees.xgboost_tree import XGBoostTree



class XGBoostRegressor(XGBoostBase):
    
    def __init__(
        self,
        n_base_models: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.2,
        lamda: float = 0.0,
        gamma: float = 0.0,
        initial_pred: float = 0.0,
        min_samples_leaf: int = 1,
        pert_radius: Union[float, np.ndarray] = 0,
        rob_alpha: float = 1.0,
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
        """
        
        super().__init__(
            loss = XGBoostMSELoss(),
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
        
        if weights is not None:
            
            weights = weights * len(weights)/sum(weights)
        
        predicted_vals = np.array([self.initial_pred] * len(annotations))
        
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
            
            base_model.fit(data,
                            annotations,
                            weights,
                            predicted_vals)
                
            predicted_vals = predicted_vals + base_model.predict(data)
            
            if base_model._root_node.is_leaf:
                break
            
            self._base_models_.append(base_model)

            
    def predict(self, data: np.ndarray):

        """
        Predicts classes for X. 
        Currently only binary classification supported.

        Args:
            data:
                Array of shape (n_samples, n_features). The input samples.
            threshold:
                Float, that describes the probability threshold for 
                classification
        Returns:
            The predicted values, array of shape (n_samples,).
        """
        
        predicted_vals = np.array([self.initial_pred]*len(data))
        
        for base_model in self._base_models_:
            
            predicted_vals += base_model.predict(data)
            
        return predicted_vals
        