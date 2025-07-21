"""
The Loss classes implementing various XGBoost loss functions.
"""

from abc import ABC
from typing import Optional

import numpy as np


class XGBoostLoss(ABC):
    """
    Generic loss for XGBoost Ensembles
    """
    
    def __call__(self,
                 prediction: np.array,
                 annotations: np.array,
                 weights: Optional[np.array] = None) -> np.array:

        """
        Wrapper for loss.
        """

        return self.loss(prediction, annotations, weights)
    
    @staticmethod
    def _calculate_estimated_loss(
            sum_of_first_derivative: float,
            sum_of_second_derivative: float, 
            lamda: float
        ) -> float:
        """
        Calculate loss estimates of the taylor series expansion given the 
        sum of first and second derivatives. 
        """
        
        return -(sum_of_first_derivative ** 2) / (sum_of_second_derivative + lamda)
    
    def _calculate_optimal_value_slow(
            self,  
            annotations: np.array,
            previous_probabilities: np.array,
            weights: Optional[np.array] = None,
            lamda: float = 0.0) -> float:
        
        """
        Calculate optimal values given the weights, annotations, and previous probabilities
        """
        
        sum_of_first_derivative = np.sum(
            self._calculate_first_derivative(annotations=annotations, pred_prob_prev=previous_probabilities)
            )
        
        sum_of_second_derivative = np.sum(
            self._calculate_second_derivative(annotations=annotations, pred_prob_prev=previous_probabilities)
            )
        
        return -(sum_of_first_derivative) / (sum_of_second_derivative + lamda)
    

    def _calculate_optimal_value(
            self,
            sum_of_first_derivative: float,
            sum_of_second_derivative: float, 
            lamda: float = 0.0
        ) -> float:
        
        """
        Calculate optimal values given the sum of first and second derivatives. 
        Two options of the function provided for greater efficiency. 
        """
        
        return -(sum_of_first_derivative) / (sum_of_second_derivative + lamda)
    
    @staticmethod
    def _calculate_first_derivative(
        annotations: np.array,
        **kwargs
    ):
        
        raise NotImplementedError()
    
    @staticmethod
    def _calculate_second_derivative(
        annotations: np.array,
        **kwargs
    ):
        
        raise NotImplementedError()


class XGBoostLogisticLoss(XGBoostLoss):
    """
    Logistic binary classification loss for XGBoost
    """
    
    def loss(self,
             prediction: np.array,
             annotations: np.array,
             weights: Optional[np.array] = None) -> np.array:
        
        return weights*((annotations*np.log(prediction)) + ((1 - annotations)*np.log(1 - prediction)))
    
    @staticmethod
    def _calculate_first_derivative(
        annotations: np.array,
        pred_prob_prev: np.array,
    ) -> np.array:
        
        return -(annotations-pred_prob_prev)
    
    @staticmethod
    def _calculate_second_derivative(
        annotations: np.array,
        pred_prob_prev: np.array,
    ) -> np.array:
        
        return pred_prob_prev*(1-pred_prob_prev)
    
    @staticmethod
    def convert_probability_to_logits(
        probabilities: np.array
        ) -> np.array:
        
        return np.log(probabilities / (1 - probabilities))
    
class XGBoostMSELoss(XGBoostLoss):
    """
    Mean squared error loss for XGBoost
    """
    
    def loss(self,
             prediction: np.array,
             annotations: np.array,
             weights: Optional[np.array] = None) -> np.array:
        
        return 0.5*((weights*(annotations-prediction))**2)
    
    @staticmethod
    def _calculate_first_derivative(
        annotations: np.array,
        pred_prob_prev: np.array,
    ) -> np.array:
        
        return -(annotations-pred_prob_prev)
    
    @staticmethod
    def _calculate_second_derivative(
        annotations: np.array,
        pred_prob_prev: np.array,
    ) -> np.array:
        
        return np.array([1.0]*len(annotations))
    