"""
XGBoost Base Class Implementation using Taylor-Series 
Approximation of Loss and Regularisation

"""

from abc import abstractmethod
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from typing import Union

from ..decision_trees.xgboost_tree import XGBoostTree

class XGBoostBase:
    
    def __init__(
        self,
        loss,
        n_base_models: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.2,
        lamda: float = 0.0,
        gamma: float = 0.0,
        initial_pred: float = 0.5,
        min_samples_leaf: int = 1,
        pert_radius: Union[float, np.ndarray] = 0,
        rob_alpha: float = 1.0,
    ):

        """
        Args:
            loss:
                The loss function to be used for training the model
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

        self.n_base_models = n_base_models

        self.loss = loss
        
        base_model = XGBoostTree(loss_func=loss, 
                                 max_depth=max_depth, 
                                 lamda=lamda, 
                                 gamma=gamma, 
                                 min_samples_leaf=min_samples_leaf,
                                 pert_radius=pert_radius,
                                 rob_alpha=rob_alpha,
                                 learning_rate=learning_rate
                                 )
        
        self._pert_radius = float(pert_radius) if isinstance(pert_radius, int) else pert_radius

        self._base_model = base_model
        self.num_base_models = n_base_models
        
        self.learning_rate = learning_rate

        self._base_models_ = []
        
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.gamma = gamma
        
        self.initial_pred = initial_pred
        
        self.one_hot_preprocesser = None
        
        
    def save_model(self, filepath: str):

        """
        Saves the model in JSON format.

        The format follows the JSON format as used in the XGBoost library.
        However, values that are not used by Venus may be missing.

        Args:
            filepath:
                The path of the savefile
        """
        tree_dicts = [
            dt._root_node.to_dict() for dt in self._base_models_
        ]

        with open(filepath, "w", buffering=1) as f:
            json.dump(tree_dicts, f, indent=2)
    
    @abstractmethod
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
        pass
            
    @abstractmethod
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
        
        pass
    
    def fit_one_hot_encoder(self, data: pd.DataFrame):
        """
        Fits the one hot encoder to encode categorical features in the data.

        Args:
            data:
                An input dataframe of shape (n_samples, n_features).

        Returns:
            The encoded data of shape (n_samples, n_features + n_categorical).
        """
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Expected data to be a pandas DataFrame for categorical encoding")
        
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
        
        transformers = []
        
        for col in data.columns:
            if col in categorical_cols:
                
                name = f"one_hot_{col}"
                
                transformers.append(
                    (name, OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"), [col])
                )
            else:
                
                name = f"numeric_{col}"
                
                transformers.append((name, "passthrough", [col]))

        self.one_hot_preprocesser = ColumnTransformer(transformers=transformers)
        
        processed_data = self.one_hot_preprocesser.fit_transform(data)
        
        self.feature_index_mapping = self._get_feature_index_mapping(
                                            self.one_hot_preprocesser, data
                                        )
        
        self.num_features = processed_data.shape[1]
        
        return processed_data
    
    @staticmethod
    def _get_feature_index_mapping(preprocessor, input_df):
        feature_mapping = {}
        current_index = 0

        for name, transformer, columns in preprocessor.transformers_:
            
            if "numeric" in name:
            
                cols = columns if isinstance(columns, list) else [columns]
                for col in cols:
                    feature_mapping[col] = [current_index]
                    current_index += 1
            
            elif "one_hot" in name:
                
                if hasattr(transformer, "get_feature_names_out"):
                    output_features = transformer.get_feature_names_out(columns)
                else:
                    output_features = [f"{columns}_{i}" for i in range(transformer.transform(input_df[columns]).shape[1])]
                
                for col in columns:
                    
                    num_cols = sum(1 for f in output_features if f.startswith(col + "_"))
                    feature_mapping[col] = list(range(current_index, current_index + num_cols))
                    current_index += num_cols

        return feature_mapping
    
    def set_perturbation_radius(self, 
                                pert_radius: Union[float, np.ndarray] = None, 
                                pert_radius_dict: dict = None):
        
        if pert_radius is not None:
            self.pert_radius = pert_radius
            
        elif pert_radius_dict is not None:
            
            if self.feature_index_mapping is None:
                raise ValueError("Feature index mapping not available. Call fit_one_hot_encoder first.")
            
            pert_radius = np.zeros(self.num_features)
            for feature, indices in self.feature_index_mapping.items():
                for ind in indices:
                    if feature in pert_radius_dict:
                        pert_radius[ind] = pert_radius_dict[feature]
                    else:
                        pert_radius[ind] = 0.0
                        
            self.pert_radius = pert_radius
        
        else:
            raise ValueError("Either pert_radius or pert_radius_dict must be provided.")
            
    
    def transform_one_hot(self, data: pd.DataFrame):
        """
        Transforms the input data using the one hot encoder.

        Args:
            data:
                An input dataframe of shape (n_samples, n_features).

        Returns:
            The encoded data of shape (n_samples, n_features + n_categorical).
        """
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Expected data to be a pandas DataFrame for categorical encoding")
        
        if self.one_hot_preprocesser is None:
            raise ValueError("One hot encoder not fitted. Call fit_one_hot_encoder first.")
        
        processed_data = self.one_hot_preprocesser.transform(data)
        
        return processed_data
    
    def save_one_hot_encoder(self, filepath: str):
        """
        Saves the one hot encoder to disk.

        Args:
            filepath:
                The path of the savefile
        """
        
        if self.one_hot_preprocesser is None:
            raise ValueError("One hot encoder not fitted. Call fit_one_hot_encoder first.")
        
        with open(filepath, "wb") as f:
            pickle.dump(self.one_hot_preprocesser, f)
            
    def load_one_hot_encoder(self, filepath: str):
        """
        Loads the one hot encoder from disk.

        Args:
            filepath:
                The path of the savefile
        """
        
        with open(filepath, "rb") as f:
            self.one_hot_preprocesser = pickle.load(f)
    
    