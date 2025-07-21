"""
Tests for XGBoost classification and regression
"""

import unittest
import numpy as np
from robust_xgboost.models.ensembles.xgboost_classifier import XGBoostClassifier 
from robust_xgboost.models.ensembles.xgboost_regressor import XGBoostRegressor

class TestXGBoostClassifier(unittest.TestCase):
    
    def test_classifier_initialization(self):
        
        model = XGBoostClassifier(
            n_base_models=10,
            max_depth=3,
            learning_rate=0.1,
            lamda=0.1,
            gamma=0.1,
            initial_pred=0.5,
            pert_radius=0.1,
            rob_alpha=0.5
        )
        self.assertEqual(model.n_base_models, 10)
        self.assertEqual(model.max_depth, 3)
        self.assertEqual(model.learning_rate, 0.1)
        self.assertEqual(model.lamda, 0.1)
        self.assertEqual(model.gamma, 0.1)
        self.assertEqual(model.initial_pred, 0.5)
        self.assertEqual(model._pert_radius, 0.1)
        self.assertEqual(model._base_model.rob_alpha, 0.5)
    
    def test_classifier_fit_one_tree(self):
        
        model = XGBoostClassifier(
            n_base_models=1,
            max_depth=1,
            learning_rate=1.0,
            lamda=0.0,
            gamma=0.0,
            initial_pred=0.5,
            min_samples_leaf=1,
            pert_radius=0.0,
            rob_alpha=1.0
        )
        
        x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1])
        
        model.fit(x, y)
        
        expected_split_feature = 0
        expected_split_value = 7.5
        
        self.assertEqual(model._base_models_[0]._root_node._split_feature_idx, expected_split_feature)
        self.assertEqual(model._base_models_[0]._root_node._split_value, expected_split_value)
        
        self.assertEqual(model._base_models_[0]._root_node._left_child.prediction, -0.857143)
        self.assertEqual(model._base_models_[0]._root_node._right_child.prediction, 2.0)
        self.assertEqual(model._base_models_[0]._root_node._left_child.is_leaf, True)
        self.assertEqual(model._base_models_[0]._root_node._right_child.is_leaf, True)
        
        expected_preds = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1]).astype(float)
        expected_proba = np.array([0.2979366 , 0.2979366 , 0.2979366 , 0.2979366 , 0.2979366 ,
                                    0.2979366 , 0.2979366 , 0.88079708, 0.88079708, 0.88079708])
        
        self.assertTrue(np.allclose(model.predict(x), expected_preds, rtol=1e-5, atol=1e-8))
        
        self.assertTrue(np.allclose(model.predict_proba(x), expected_proba, rtol=1e-5, atol=1e-8))
        
    def test_classifier_100_trees(self):
        
        model = XGBoostClassifier(
            n_base_models=100,
            max_depth=1,
            learning_rate=1.0,
            lamda=0.0,
            gamma=0.0,
            initial_pred=0.5,
            min_samples_leaf=1,
            pert_radius=0.0,
            rob_alpha=1.0
        )
        
        x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1])
        
        model.fit(x, y)
        
        expected_preds = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1])
    
        print(model.predict_proba(x))
        print(model.predict_proba(x) - expected_preds)
        
        self.assertTrue(np.allclose(model.predict(x), expected_preds, rtol=1e-5, atol=1e-8))
        self.assertTrue(np.allclose(model.predict_proba(x), expected_preds, rtol=1e-5, atol=1e-8))
        
class TestXGBoostRegressor(unittest.TestCase):
    
    def test_regressor_250_trees(self):
        
        x = np.arange(1,1000)
        y = np.arange(1,1000) / 6

        x = np.array(x).reshape(-1, 1)
        y = np.array(y)

        model = XGBoostRegressor(
            n_base_models=250,
            initial_pred=0.0,
            max_depth=4,
            lamda=0.0,
            learning_rate=1
        )
        
        model.fit(x, y)
        
        self.assertTrue(np.allclose(model.predict(x), y, rtol=1e-5, atol=1e-8))

    def test_pruning_early_stopping(self):
        
        x = np.arange(1,1000)
        y = np.arange(1,1000) / 6

        x = np.array(x).reshape(-1, 1)
        y = np.array(y)

        model = XGBoostRegressor(
            n_base_models=1000,
            initial_pred=0.0,
            max_depth=10,
            lamda=0.0,
            learning_rate=1
        )
        
        model.fit(x, y)
        
        self.assertTrue(np.allclose(model.predict(x), y, rtol=1e-5, atol=1e-8))
        self.assertEqual(len(model._base_models_), 1)