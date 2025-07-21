"""
RobustXGBoost: A robust implementation of XGBoost with enhanced decision tree algorithms.
"""

__version__ = "1.0.0"
__author__ = "Atri Sharma"

# Import main classes for easy access
from .models.ensembles import XGBoostClassifier, XGBoostRegressor
from .models.decision_trees import DecisionTree, DTNode, XGBoostDTNode, XGBoostTree

__all__ = [
    "XGBoostClassifier",
    "XGBoostRegressor", 
    "DecisionTree",
    "DTNode",
    "XGBoostDTNode",
    "XGBoostTree"
]
