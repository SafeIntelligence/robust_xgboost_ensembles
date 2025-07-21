"""
Unittests for the XGBoost Tree and Robust loss
"""

import unittest
import numpy as np
from robust_xgboost.models.decision_trees.xgboost_tree import XGBoostTree
from robust_xgboost.models.decision_trees.losses import XGBoostLogisticLoss, XGBoostMSELoss

class TestXGBoostTree(unittest.TestCase):
    
    def setUp(self):
        self.logistic_loss = XGBoostLogisticLoss()
        self.mse_loss = XGBoostMSELoss()
        
    def test_classification_vanilla_xgboost_training_depth_1(self):
        
        base_model = XGBoostTree(loss_func=self.logistic_loss, 
                                 max_depth=1, 
                                 lamda=0, 
                                 gamma=0, 
                                 learning_rate=1.0
                                 )
        
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1]

        x = np.array(x).reshape(-1, 1)
        y = np.array(y)
        
        base_model.fit(x, y)
        
        expected_split_feature = 0
        expected_split_value = 7.5
        
        expected_preds = np.ones_like(y).astype(float)
        
        expected_preds[x.reshape(-1) < expected_split_value] = -0.857143
        expected_preds[x.reshape(-1) >= expected_split_value] = 2
        
        assert base_model._root_node._split_feature_idx == expected_split_feature
        assert base_model._root_node._split_value == expected_split_value
        assert np.allclose(base_model.predict(x), expected_preds, rtol=1e-5, atol=1e-8)
        
    def test_classification_vanilla_xgboost_training_depth_2(self):
        
        base_model = XGBoostTree(loss_func=self.logistic_loss, 
                                 max_depth=2, 
                                 lamda=0, 
                                 gamma=0, 
                                 learning_rate=1.0
                                 )
        
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1]

        x = np.array(x).reshape(-1, 1)
        y = np.array(y)
        
        base_model.fit(x, y)
        
        expected_split_feature = 0
        expected_split_value_root = 7.5
        expected_split_value_left = 4.5
        
        expected_preds = np.array([0, 0, 0, 0, -2, -2, -2, 2, 2, 2]).astype(float)
        
        assert base_model._root_node._split_feature_idx == expected_split_feature
        assert base_model._root_node._split_value == expected_split_value_root
        assert base_model._root_node._left_child._split_value == expected_split_value_left
        assert np.allclose(base_model.predict(x), expected_preds, rtol=1e-5, atol=1e-8)
            
    def test_regression_vanilla_xgboost_training_depth_1(self):
        
        base_model = XGBoostTree(loss_func=self.mse_loss, 
                                 max_depth=1, 
                                 lamda=0, 
                                 gamma=0, 
                                 learning_rate=1.0,
                                 )
        
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1]

        x = np.array(x).reshape(-1, 1)
        y = np.array(y)
        
        base_model.fit(x, y, previous_predictions=0.0)
        
        expected_split_feature = 0
        expected_split_value = 7.5
        
        expected_preds = np.ones_like(y).astype(float)
        
        expected_preds[x.reshape(-1) < expected_split_value] = 0.285714
        expected_preds[x.reshape(-1) >= expected_split_value] = 1.0
        
        assert base_model._root_node._split_feature_idx == expected_split_feature
        assert base_model._root_node._split_value == expected_split_value
        assert np.allclose(base_model.predict(x), expected_preds, rtol=1e-5, atol=1e-8)
    
    def test_regression_vanilla_xgboost_training_depth_2(self):
        
        base_model = XGBoostTree(loss_func=self.mse_loss, 
                                 max_depth=2, 
                                 lamda=0, 
                                 gamma=0, 
                                 learning_rate=1.0,
                                 )
        
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1]

        x = np.array(x).reshape(-1, 1)
        y = np.array(y)
        
        base_model.fit(x, y, previous_predictions=0.0)    
        
        
        expected_split_feature = 0
        expected_split_value_root = 7.5
        expected_split_value_left = 4.5
        
        expected_preds = np.array([0.5, 0.5, 0.5, 0.5, 0, 0, 0, 1, 1, 1]).astype(float)
        
        assert base_model._root_node._split_feature_idx == expected_split_feature
        assert base_model._root_node._split_value == expected_split_value_root
        assert base_model._root_node._left_child._split_value == expected_split_value_left
        assert np.allclose(base_model.predict(x), expected_preds, rtol=1e-5, atol=1e-8)
            
    def test_classification_vanilla_xgboost_2_dims(self):
        
        x = [[1, 1], [1, 2], [3, 1], [3, 2], [5, 1], [5, 2], [7, 1], [7, 2], [9, 1], [9, 2]]
        y = [0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
        x = np.array(x).astype(float)
        y = np.array(y)

        base_model = XGBoostTree(loss_func=self.logistic_loss, 
                    max_depth=1, 
                    lamda=0, 
                    gamma=0, 
                    learning_rate=1.0,
                    )

        base_model.fit(x, y)
        
        expected_split_feature = 1
        expected_split_value = 1.5
        
        expected_preds = np.array([-1.2, 1.2, -1.2, 1.2, -1.2, 1.2, -1.2, 1.2, -1.2, 1.2])
        
        assert base_model._root_node._split_feature_idx == expected_split_feature
        assert base_model._root_node._split_value == expected_split_value
        assert np.allclose(base_model.predict(x), expected_preds, rtol=1e-5, atol=1e-8)
        
    def test_classification_robust_vanilla_xgboost_2_dims(self):
        
        x = [[1, 1], [1, 2], [3, 1], [3, 2], [5, 1], [5, 2], [7, 1], [7, 2], [9, 1], [9, 2]]
        y = [0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
        x = np.array(x).astype(float)
        y = np.array(y)

        base_model = XGBoostTree(loss_func=self.logistic_loss, 
                    max_depth=1, 
                    lamda=0, 
                    gamma=0, 
                    learning_rate=1.0,
                    pert_radius=1.0
                    )

        base_model.fit(x, y)
        
        expected_split_feature = 0
        expected_split_value = 6
        
        expected_preds = np.array([-0.285714, -0.285714, -0.285714, -0.285714, -0.285714, -0.285714, 2/3, 2/3, 2/3, 2/3])
        
        assert base_model._root_node._split_feature_idx == expected_split_feature
        assert base_model._root_node._split_value == expected_split_value
        assert np.allclose(base_model.predict(x), expected_preds, rtol=1e-5, atol=1e-8)
        
    def test_regression_vanilla_xgboost_training_depth_4_complex(self):
        
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        x = np.array(x).reshape(-1, 1)
        y = np.array(y)
        
        base_model = XGBoostTree(loss_func=self.mse_loss, 
                                 max_depth=4, 
                                 lamda=0, 
                                 gamma=0, 
                                 learning_rate=1.0,
                                 )
        
        base_model.fit(x, y, previous_predictions=0.0)
        
        preds = base_model.predict(x)
        
        assert np.allclose(preds, y, rtol=1e-5, atol=1e-8)
        
    def test_regression_vanilla_xgboost_training_depth_10_complex(self):
        
        x = np.arange(1, 1000)
        y = np.arange(1, 1000)/16
        
        x = np.array(x).reshape(-1, 1)
        y = np.array(y)
        
        base_model = XGBoostTree(loss_func=self.mse_loss, 
                                 max_depth=10, 
                                 lamda=0, 
                                 gamma=0, 
                                 learning_rate=1.0,
                                 )
        
        base_model.fit(x, y, previous_predictions=0.0)
        
        preds = base_model.predict(x)
        
        assert np.allclose(preds, y, rtol=1e-5, atol=1e-8)