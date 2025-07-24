"""
Weak Estimator to be used in an XGBoost Ensemble

Inspired from the original XGBoost paper by Chen et. al. 2016: 
https://arxiv.org/pdf/1603.02754
"""
from typing import Optional, Union, Tuple

import numpy as np

from .losses import XGBoostLogisticLoss, XGBoostLoss
from .losses.robust_loss_xgboost import compute_robust_loss

from .xgboost_dt_node import XGBoostDTNode
from .decision_tree import DecisionTree
from numba import njit, prange

np.seterr(all='raise')

# Define a small tolerance to prevent division by zero
TOL = 1e-9


@njit(error_model='numpy', cache=True)
def optimized_update(
    val: float,
    index: int,                       
    sorted_vals: np.ndarray,    
    sorted_indices: np.ndarray,       
    radius: float,              
    ambiguous_index_start_counter: int,
    ambiguous_index_end_counter: int,  
    previous_ambiguous_index_start_counter: int, 
    previous_ambiguous_index_end_counter: int,   
    number_of_points: int,     
    first_derivatives_at_node: np.ndarray, 
    second_derivatives_at_node: np.ndarray, 
    sum_of_first_derivative_left: float,    
    sum_of_second_derivative_left: float,   
    sum_of_first_derivative: float,         
    sum_of_second_derivative: float,        
    sum_of_unambiguous_first_derivative_left: float,   
    sum_of_unambiguous_second_derivative_left: float,  
    max_sum_of_ambiguous_first_derivative: float,      
    min_sum_of_ambiguous_first_derivative: float,      
    sum_of_ambiguous_first_derivative: float,          
    sum_of_ambiguous_second_derivative: float,
    min_gi_amb: float,
    max_gi_amb: float,
    min_hi_amb: float,
    max_hi_amb: float,
) -> Tuple[int, int, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float]:
    
    """
    Numba optimized function to update the values of the amiguous indices, sums of derivatives and
    other relevant variables when a new split is considered.
    
    Args:
        val: float
            The next split value to consider
        index: int
            The index of the current feature.
        sorted_vals: np.ndarray
            The sorted values of the feature to split on.
        sorted_indices: np.ndarray
            The sorted indices of the training data.
        radius: float
            The perturbation radius on the feature.
        ambiguous_index_start_counter: int
            The current index of the first ambiguous point i.e. the first point 
            that lies within the perturbation radius of the split value.
        ambiguous_index_end_counter: int
            The current index of the last ambiguous point i.e. the last point
            that lies within the perturbation radius of the split value.
        previous_ambiguous_index_start_counter: int
            The previous index of the first ambiguous point.
        previous_ambiguous_index_end_counter: int
            The previous index of the last ambiguous point.
        number_of_points: int
            The total number of points in the training data of the current node.
        first_derivatives_at_node: np.ndarray
            The first derivatives (g_i) of the loss function at the current node.
        second_derivatives_at_node: np.ndarray
            The second derivatives (h_i) of the loss function at the current node.
        sum_of_first_derivative_left: float
            The current sum of first derivatives on the left child node.
        sum_of_second_derivative_left: float
            The current sum of second derivatives on the left child node.
        sum_of_unambiguous_first_derivative_left: float
            The current sum of first derivatives on the left child node 
            excluding ambiguous points.
        sum_of_unambiguous_second_derivative_left: float
            The current sum of second derivatives on the left child node 
            excluding ambiguous points.
        max_sum_of_ambiguous_first_derivative: float
            The maximum possible sum of the first derivatives of the 
            ambiguous points. This is taken by adding all positive
            values of the first derivative.
        min_sum_of_ambiguous_first_derivative: float
            The minimum possible sum of the first derivatives of the
            ambiguous points. This is taken by adding all negative
            values of the first derivative.
        sum_of_ambiguous_first_derivative: float
            The current sum of the first derivatives of the ambiguous points.
        sum_of_ambiguous_second_derivative: float
            The current sum of the second derivatives of the ambiguous points.
        min_gi_amb: float
            The minimum value of the first derivative of the ambiguous points.
        max_gi_amb: float
            The maximum value of the first derivative of the ambiguous points.
        min_hi_amb: float
            The minimum value of the second derivative of the ambiguous points.
        max_hi_amb: float
            The maximum value of the second derivative of the ambiguous points.
            
    Returns:
        
        tuple: The updated values of input variables on the next split. 
    """
        
    while (ambiguous_index_start_counter < number_of_points) and (sorted_vals[ambiguous_index_start_counter] <= val - radius):
        ambiguous_index_start_counter += 1
    
    while (ambiguous_index_end_counter < number_of_points) and (sorted_vals[ambiguous_index_end_counter] <= val + radius):
        ambiguous_index_end_counter += 1
    
    first_derivative_at_point = first_derivatives_at_node[index]
    second_derivative_at_point = second_derivatives_at_node[index]
    

    sum_of_first_derivative_left += first_derivative_at_point
    sum_of_second_derivative_left += second_derivative_at_point
    

    sum_of_first_derivative_right = sum_of_first_derivative - sum_of_first_derivative_left
    sum_of_second_derivative_right = sum_of_second_derivative - sum_of_second_derivative_left
    

    ambiguous_indices_removed = sorted_indices[previous_ambiguous_index_start_counter:ambiguous_index_start_counter]
    ambiguous_indices_added = sorted_indices[previous_ambiguous_index_end_counter:ambiguous_index_end_counter]
    
    recalculate = False
    

    for i in range(ambiguous_index_start_counter - previous_ambiguous_index_start_counter):
        ambiguous_index = ambiguous_indices_removed[i]
        fd = first_derivatives_at_node[ambiguous_index]
        sd = second_derivatives_at_node[ambiguous_index]
        
        sum_of_unambiguous_first_derivative_left += fd
        sum_of_unambiguous_second_derivative_left += sd
        
        sum_of_ambiguous_first_derivative -= fd
        sum_of_ambiguous_second_derivative -= sd
        
        if fd > 0.0:
            max_sum_of_ambiguous_first_derivative -= fd
        else:
            min_sum_of_ambiguous_first_derivative -= fd
            
        
        if fd == max_gi_amb:
            recalculate = True
            continue
            
        if fd == min_gi_amb:
            recalculate = True
            continue
            
        if sd == max_hi_amb:
            recalculate = True
            continue
            
        if sd == min_hi_amb:
            recalculate = True
            continue           
            
    for i in range(ambiguous_index_end_counter - previous_ambiguous_index_end_counter):
        ambiguous_index = ambiguous_indices_added[i]
        fd = first_derivatives_at_node[ambiguous_index]
        sd = second_derivatives_at_node[ambiguous_index]
        
        sum_of_ambiguous_first_derivative += fd
        sum_of_ambiguous_second_derivative += sd
        
        if fd > 0.0:
            max_sum_of_ambiguous_first_derivative += fd
        else:
            min_sum_of_ambiguous_first_derivative += fd
            
        if fd > max_gi_amb:
            max_gi_amb = fd
            
        if fd < min_gi_amb:
            min_gi_amb = fd
            
        if sd > max_hi_amb:
            max_hi_amb = sd
            
        if sd < min_hi_amb:
            min_hi_amb = sd
       
        
    if recalculate:
    
        for ind in sorted_indices[ambiguous_index_start_counter:ambiguous_index_end_counter]:
            fd = first_derivatives_at_node[ind]
            sd = second_derivatives_at_node[ind]
            
            if fd > max_gi_amb:
                max_gi_amb = fd
                
            if fd < min_gi_amb:
                min_gi_amb = fd
                
            if sd > max_hi_amb:
                max_hi_amb = sd
                
            if sd < min_hi_amb:
                min_hi_amb = sd
    
    sum_of_unambiguous_first_derivative_right = sum_of_first_derivative - sum_of_ambiguous_first_derivative - sum_of_unambiguous_first_derivative_left
    sum_of_unambiguous_second_derivative_right = sum_of_second_derivative - sum_of_ambiguous_second_derivative - sum_of_unambiguous_second_derivative_left
    
    return (
        ambiguous_index_start_counter,
        ambiguous_index_end_counter,
        sum_of_first_derivative_left,
        sum_of_second_derivative_left,
        sum_of_first_derivative_right,
        sum_of_second_derivative_right,
        sum_of_unambiguous_first_derivative_left,
        sum_of_unambiguous_second_derivative_left,
        max_sum_of_ambiguous_first_derivative,
        min_sum_of_ambiguous_first_derivative,
        sum_of_ambiguous_first_derivative,
        sum_of_ambiguous_second_derivative,
        sum_of_unambiguous_first_derivative_right,
        sum_of_unambiguous_second_derivative_right,
        min_gi_amb,
        max_gi_amb,
        min_hi_amb,
        max_hi_amb
    )
    
@njit(error_model='numpy', cache=True)
def calculate_estimated_loss_numba(sum_of_first_derivative: float, sum_of_second_derivative: float, lamda: float) -> float:
    
    """
    Compute the estimated loss (negative score function) at a given node, derived from
    the Taylor expansion of the loss in XGBoost.
    
    Args:
        sum_of_first_derivative: float
            The sum of first derivatives at the node.
        sum_of_second_derivative: float
            The sum of second derivatives at the node.
        lamda: float
            The regularisation parameter.
    
    Returns:
        float: The estimated loss at the node.
    
    """
    
    return -(sum_of_first_derivative ** 2) / (sum_of_second_derivative + lamda + TOL)

@njit(error_model='numpy', cache=True)
def calculate_optimal_value_numba(sum_of_first_derivative: float, sum_of_second_derivative: float, lamda: float) -> float:
    
    """
    Compute the optimal leaf value at a given node, derived from the Taylor 
    expansion of the loss in XGBoost.
    
    Args:
        sum_of_first_derivative: float
            The sum of first derivatives at the node.
        sum_of_second_derivative: float
            The sum of second derivatives at the node.
        lamda: float
            The regularisation parameter.
            
    Returns:
        float: The optimal leaf value at the node.
    """

    return -(sum_of_first_derivative) / (sum_of_second_derivative + lamda + TOL)
    
@njit(error_model='numpy', cache=True)
def fit_one_node_optimized(
    train_X_split: np.ndarray, 
    node_indices: np.ndarray, 
    pert_radius: np.ndarray, 
    first_derivatives_at_node: np.ndarray, 
    second_derivatives_at_node: np.ndarray, 
    lamda: float, 
    gamma: float, 
    rob_alpha: float
    ) -> Tuple[float, int, float, float, float, float, float]:
    
    """
    Numba optimized function to find the best split for a given node. 
    
    Args:
        train_X_split: np.ndarray
            The training data at the node.
        node_indices: np.ndarray
            The indices of the training data at the node.
        pert_radius: np.ndarray
            The perturbation radius for each feature.
        first_derivatives_at_node: np.ndarray
            The first derivatives of the loss function at the node.
        second_derivatives_at_node: np.ndarray
            The second derivatives of the loss function at the node.
        lamda: float
            The regularisation parameter.
        gamma: float
            The gamma parameter in XGBoost.
        rob_alpha: float
            The weightage of the robust loss in the loss function.
            
    Returns:
        tuple: The best loss reduction, best feature index, best threshold, 
        best optimal value for the left child, best optimal value for the right child,
        best cover for the left child, best cover for the right child.
    """
    
    number_of_features = train_X_split.shape[1]
    
    number_of_points = len(node_indices)
    
    best_loss_reduction = -np.inf
    best_feature_index = 0
    best_threshold = 0.0
    best_optimal_value_left_child = 0.0
    best_optimal_value_right_child = 0.0
    
    sum_of_first_derivative = np.sum(first_derivatives_at_node)
    sum_of_second_derivative = np.sum(second_derivatives_at_node)
    
    pre_split_loss_estimate = calculate_estimated_loss_numba(
        sum_of_first_derivative=sum_of_first_derivative,
        sum_of_second_derivative=sum_of_second_derivative,
        lamda=lamda
    )
        
    best_cover_left = 0.0
    best_cover_right = 0.0
    

    for feature_index in range(number_of_features):
        
        sum_of_first_derivative_left = 0.0
        sum_of_second_derivative_left = 0.0
        
        counter = 0
        
        radius = pert_radius[feature_index]
        
        sorted_indices = np.argsort(train_X_split[:, feature_index])
        sorted_vals = train_X_split[sorted_indices, feature_index]

        ambiguous_index_start_counter = 0
        ambiguous_index_end_counter = 0
        
        previous_ambiguous_index_start_counter = 0
        previous_ambiguous_index_end_counter = 0
        
        sum_of_unambiguous_first_derivative_left = 0.0
        sum_of_unambiguous_second_derivative_left = 0.0
        
        sum_of_ambiguous_first_derivative = 0.0
        sum_of_ambiguous_second_derivative = 0.0
        
        max_sum_of_ambiguous_first_derivative = 0.0
        min_sum_of_ambiguous_first_derivative = 0.0
        
        loss_reduction = -np.inf
        sum_of_first_derivative_left_adversarial = 0.0
        sum_of_second_derivative_left_adversarial = 0.0
        
        sum_of_first_derivative_right_adversarial = 0.0
        sum_of_second_derivative_right_adversarial = 0.0
        
        estimated_loss_normal_left = 0.0
        estimated_loss_normal_right = 0.0
        estimated_loss_normal = 0.0
            
        min_gi_amb = 1000
        max_gi_amb = -1000
        min_hi_amb = 1000
        max_hi_amb = -1000
        
        for i in range(0, number_of_points-1):
            
            index = sorted_indices[i]
            
            next_index = sorted_indices[i+1]
            
            if train_X_split[index, feature_index] == train_X_split[next_index, feature_index]:
                
                first_derivative_at_point = first_derivatives_at_node[index]
                second_derivative_at_point = second_derivatives_at_node[index]
                sum_of_first_derivative_left += first_derivative_at_point
                sum_of_second_derivative_left += second_derivative_at_point
                continue
            
            if radius != 0:
                
                val = (train_X_split[index, feature_index] + train_X_split[next_index, feature_index])/2
                    
                (
                    ambiguous_index_start_counter,
                    ambiguous_index_end_counter,
                    sum_of_first_derivative_left,
                    sum_of_second_derivative_left,
                    sum_of_first_derivative_right,
                    sum_of_second_derivative_right,
                    sum_of_unambiguous_first_derivative_left,
                    sum_of_unambiguous_second_derivative_left,
                    max_sum_of_ambiguous_first_derivative,
                    min_sum_of_ambiguous_first_derivative,
                    sum_of_ambiguous_first_derivative,
                    sum_of_ambiguous_second_derivative,
                    sum_of_unambiguous_first_derivative_right,
                    sum_of_unambiguous_second_derivative_right,
                    min_gi_amb,
                    max_gi_amb,
                    min_hi_amb,
                    max_hi_amb,
                ) =optimized_update(
                    val,    
                    index,                        
                    sorted_vals,   
                    sorted_indices,          
                    radius,                 
                    ambiguous_index_start_counter, 
                    ambiguous_index_end_counter,   
                    previous_ambiguous_index_start_counter, 
                    previous_ambiguous_index_end_counter, 
                    number_of_points,      
                    first_derivatives_at_node,       
                    second_derivatives_at_node,      
                    sum_of_first_derivative_left,    
                    sum_of_second_derivative_left,   
                    sum_of_first_derivative,         
                    sum_of_second_derivative,                           
                    sum_of_unambiguous_first_derivative_left,   
                    sum_of_unambiguous_second_derivative_left,  
                    max_sum_of_ambiguous_first_derivative,      
                    min_sum_of_ambiguous_first_derivative,      
                    sum_of_ambiguous_first_derivative,          
                    sum_of_ambiguous_second_derivative,          
                    min_gi_amb,
                    max_gi_amb,
                    min_hi_amb,
                    max_hi_amb,
                )
                
                optimal_p, optimal_q, optimal_value = compute_robust_loss(
                    A = sum_of_unambiguous_first_derivative_left,
                    B = sum_of_unambiguous_first_derivative_right,
                    C = sum_of_unambiguous_second_derivative_left + lamda,
                    D = sum_of_unambiguous_second_derivative_right + lamda,
                    T = sum_of_ambiguous_first_derivative,
                    Q = sum_of_ambiguous_second_derivative,
                    p_min= min_sum_of_ambiguous_first_derivative,
                    p_max= max_sum_of_ambiguous_first_derivative,
                    min_gi_amb=min_gi_amb,
                    max_gi_amb=max_gi_amb,
                    min_hi_amb=min_hi_amb,
                    max_hi_amb=max_hi_amb,
                )
                    
                
                upper_bound_estimated_loss_adv = -optimal_value
                
                sum_of_first_derivative_left_adversarial = sum_of_unambiguous_first_derivative_left + optimal_p
                sum_of_second_derivative_left_adversarial = sum_of_unambiguous_second_derivative_left + optimal_q
                
                sum_of_first_derivative_right_adversarial = sum_of_unambiguous_first_derivative_right + sum_of_ambiguous_first_derivative - optimal_p
                sum_of_second_derivative_right_adversarial = sum_of_unambiguous_second_derivative_right + sum_of_ambiguous_second_derivative - optimal_q
                
                counter += 1
                previous_ambiguous_index_start_counter = ambiguous_index_start_counter
                previous_ambiguous_index_end_counter = ambiguous_index_end_counter         
                
                if rob_alpha < 1: 
                    estimated_loss_normal_left = calculate_estimated_loss_numba(
                        sum_of_first_derivative=sum_of_first_derivative_left,
                        sum_of_second_derivative=sum_of_second_derivative_left,
                        lamda=lamda
                    )
                    
                    estimated_loss_normal_right = calculate_estimated_loss_numba(
                        sum_of_first_derivative=sum_of_first_derivative_right,
                        sum_of_second_derivative=sum_of_second_derivative_right,
                        lamda=lamda
                    )
                    
                    estimated_loss_normal = estimated_loss_normal_left + estimated_loss_normal_right
                    
                    loss_reduction = 0.5*(pre_split_loss_estimate - (rob_alpha*upper_bound_estimated_loss_adv + (1-rob_alpha)*estimated_loss_normal)) - gamma
                    
                else:
                    
                    loss_reduction = 0.5*(pre_split_loss_estimate - upper_bound_estimated_loss_adv) - gamma
                
                if loss_reduction > best_loss_reduction:
                
                    best_feature_index = feature_index
                    best_threshold = val
                    
                    best_loss_reduction = loss_reduction
                    
                    best_optimal_value_left_child = calculate_optimal_value_numba(
                        sum_of_first_derivative = sum_of_first_derivative_left_adversarial,
                        sum_of_second_derivative = sum_of_second_derivative_left_adversarial,
                        lamda = lamda
                        )
                    
                    best_optimal_value_right_child = calculate_optimal_value_numba(
                        sum_of_first_derivative = sum_of_first_derivative_right_adversarial,
                        sum_of_second_derivative = sum_of_second_derivative_right_adversarial,
                        lamda = lamda
                        )
                                        
                    best_cover_left = sum_of_second_derivative_left
                    best_cover_right = sum_of_second_derivative_right
                
                
            else:
                
                val = (train_X_split[index, feature_index] + train_X_split[next_index, feature_index])/2
                
                first_derivative_at_point = first_derivatives_at_node[index]
                second_derivative_at_point = second_derivatives_at_node[index]
                
                sum_of_first_derivative_left += first_derivative_at_point
                sum_of_second_derivative_left += second_derivative_at_point
                
                sum_of_first_derivative_right = sum_of_first_derivative - sum_of_first_derivative_left
                sum_of_second_derivative_right = sum_of_second_derivative - sum_of_second_derivative_left
                
                estimated_loss_normal_left = calculate_estimated_loss_numba(
                        sum_of_first_derivative=sum_of_first_derivative_left,
                        sum_of_second_derivative=sum_of_second_derivative_left,
                        lamda=lamda
                    )
                    
                estimated_loss_normal_right = calculate_estimated_loss_numba(
                    sum_of_first_derivative=sum_of_first_derivative_right,
                    sum_of_second_derivative=sum_of_second_derivative_right,
                    lamda=lamda
                )
                
                estimated_loss_normal = estimated_loss_normal_left + estimated_loss_normal_right
                
                loss_reduction = 0.5*(pre_split_loss_estimate -estimated_loss_normal) - gamma
                
                sum_of_first_derivative_left_adversarial = sum_of_first_derivative_left
                sum_of_second_derivative_left_adversarial = sum_of_second_derivative_left
                
                
                sum_of_first_derivative_right_adversarial = sum_of_first_derivative_right
                sum_of_second_derivative_right_adversarial = sum_of_second_derivative_right
                    
                
                if loss_reduction > best_loss_reduction:
                
                    best_feature_index = feature_index
                    best_threshold = val
                    
                    best_loss_reduction = loss_reduction
                    
                    best_optimal_value_left_child = calculate_optimal_value_numba(
                        sum_of_first_derivative = sum_of_first_derivative_left_adversarial,
                        sum_of_second_derivative = sum_of_second_derivative_left_adversarial,
                        lamda = lamda
                        )
                    
                    best_optimal_value_right_child = calculate_optimal_value_numba(
                        sum_of_first_derivative = sum_of_first_derivative_right_adversarial,
                        sum_of_second_derivative = sum_of_second_derivative_right_adversarial,
                        lamda = lamda
                        )
                    
                    
                    best_cover_left = sum_of_second_derivative_left
                    best_cover_right = sum_of_second_derivative_right
        
        
    return (
        best_loss_reduction, 
        best_feature_index, 
        best_threshold, 
        best_optimal_value_left_child, 
        best_optimal_value_right_child, 
        best_cover_left, 
        best_cover_right
        )
    

class XGBoostTree(DecisionTree):
    """
    A class implementing XGBoost Decision Trees.
    """

    def __init__(self,
                 loss_func: XGBoostLoss = XGBoostLogisticLoss(), 
                 pert_radius: Union[float, np.ndarray] = 0,
                 max_depth: int = 10,
                 min_samples_leaf: int = 1,
                 lamda: float = 0.0,
                 gamma: float = 0.0,
                 rob_alpha: float = 1.0,
                 learning_rate: float = 1.0
                 ):

        """
        Args:
            loss_func:
                The loss functions for choosing the best split (defaults to
                gini if None).
            pert_radius:
                Either a float or torch tensor indicating the perturbation
                radius for robust training. Set to 0 for standard training.
            max_depth:
                The maximum depth of the tree.
            min_samples_leaf:
                The minimum number of datapoints required to split an internal
                node.
            lamda:
                The regularisation parameter to penalise high weight values in
                XGBoost
            gamma:
                The regularisation parameter to penalise a large number of 
                leaves in an XGBoost tree
            rob_alpha:
                The weightage of the robust loss in the loss function. A value
                of 1.0 indicates that the robust loss is used exclusively.
            learning_rate:
                The learning rate of the XGBoost model.
        """

        
        self.gamma = gamma
        self.lamda = lamda

        super().__init__(loss_func, pert_radius, max_depth, min_samples_leaf)
        self._fp_error_eps = 1e-5

        self.rob_alpha = rob_alpha
        self.learning_rate = learning_rate
    
    
    def prune(self, child_nodes: Optional[list] = None) -> None:
        """
        Prunes nodes with negative loss reduction by removing their children.
        
        Args:
            child_nodes: List of nodes to check for pruning. Defaults to leaf nodes.
        """
        if child_nodes is None:
            child_nodes = self.leaf_nodes
            
        pruned = False
        for node in child_nodes:
            parent_node = node.parent_node
            
            if parent_node is None:
                continue
            
            if round(getattr(parent_node, 'node_loss_reduction', 0), 9) <= 0:
                
                parent_node._left_child = None 
                parent_node._right_child = None
                parent_node._split_feature_idx = None
                pruned = True
                
        if pruned:
            new_leaves = self.leaf_nodes
            self.prune(child_nodes=new_leaves)

    def _init_root_node(self,
                        data: np.ndarray,
                        annotations: np.ndarray,
                        weights: Optional[np.ndarray] = None,
                        previous_predictions: Optional[Union[float, np.ndarray]] = 0.5
                        ) -> None:

        """
        Initialises the root node.

        Args:
            data:
                The input data of shape (n_datapoints, n_features).
            annotations:
                The targets for the input data.
            weights:
                The weights for the input data.
            previous_predictions:
                The previous predictions made in the ensemble.
        """
        if isinstance(previous_predictions, float):
            previous_predictions = np.array([previous_predictions]*len(annotations))
        
        initial_prediction = self._loss_func._calculate_optimal_value_slow(
                                    annotations=annotations,
                                    previous_probabilities=previous_predictions,
                                    weights=weights,
                                    lamda=self.lamda,
                                    )

        # Default parameters to predict most common class.
        self._root_node = XGBoostDTNode(
                    prediction=initial_prediction, 
                    depth=0,
                    all_data=data, 
                    all_annotations=annotations, 
                    all_weights=weights,
                    all_previous_predictions=previous_predictions,
                    training_data_indices=np.array(list(range(len(data))))
                    )
        
        self._root_node.parent_node = None
        
        self._first_derivatives = self._loss_func._calculate_first_derivative(
            annotations=annotations,
            pred_prob_prev=previous_predictions
        )
        
        self._second_derivatives = self._loss_func._calculate_second_derivative(
            annotations=annotations,
            pred_prob_prev=previous_predictions
        )
        
        if weights is not None:
            
            self._first_derivatives = weights * self._first_derivatives
            self._second_derivatives = weights * self._second_derivatives
        

    def fit(self,
            data: np.ndarray,
            annotations: np.ndarray,
            weights: Optional[np.ndarray] = None,
            previous_predictions: Optional[Union[float, np.ndarray]] = 0.5) -> None:

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
                The previous predictions made in the ensemble.

        Returns:
            The fitted model.
        """
        
        super().fit(data, annotations, weights, previous_predictions)
        self.prune()
        
        return 
    
    def _fit_one_node(self, node: XGBoostDTNode) -> Optional[Tuple[XGBoostDTNode, XGBoostDTNode]]:

        """
        Fits the best split by brute force looping through all possible splits.

        Args:
            node:
                The node to fit the best split for.

        Returns:
            The left and right child nodes.
        """
        
        
        train_X_split, train_y_split, weights, prev_pred_prob_split = node.train_set
        node_indices = node.data_indices
        
        number_of_features = train_X_split.shape[1]
            
        pert_radius = self._pert_radius
        
        if isinstance(pert_radius, float):
            pert_radius = np.array([pert_radius]*number_of_features)
            
        first_derivatives_at_node = self._first_derivatives[node_indices]
        second_derivatives_at_node = self._second_derivatives[node_indices]
        
        # n_datapoints = train_X_split.shape[0]
        (best_loss_reduction, 
         best_feature_index, 
         best_threshold, 
         best_optimal_value_left_child, 
         best_optimal_value_right_child, 
         best_cover_left, 
         best_cover_right
            )=fit_one_node_optimized(
            train_X_split=train_X_split, 
            node_indices=node_indices, 
            pert_radius=pert_radius, 
            first_derivatives_at_node=first_derivatives_at_node, 
            second_derivatives_at_node=second_derivatives_at_node, 
            lamda=self.lamda, 
            gamma=self.gamma, 
            rob_alpha=self.rob_alpha)
            
        node.node_loss_reduction = best_loss_reduction
        
        # Calculate pre-split loss for debugging
        sum_of_first_derivative = np.sum(first_derivatives_at_node)
        sum_of_second_derivative = np.sum(second_derivatives_at_node)
        pre_split_loss_estimate = -(sum_of_first_derivative ** 2) / (sum_of_second_derivative + self.lamda + TOL)
        
        left_mask = train_X_split[:,best_feature_index] < best_threshold 
        right_mask = train_X_split[:,best_feature_index] >= best_threshold
        
        
        node.encode_split(
            split_feature_idx= best_feature_index,
            split_value=best_threshold,
            indices_left=node_indices[left_mask],
            indices_right=node_indices[right_mask],
            left_child_prediction=round(best_optimal_value_left_child, 6)*self.learning_rate,
            right_child_prediction=round(best_optimal_value_right_child, 6)*self.learning_rate,
            left_cover=best_cover_left,
            right_cover=best_cover_right
        )
            
        return node.children
                             
    def clone(self) -> 'XGBoostTree':

        """
        Returns a decision tree with same hyperparameters but without any
        fitted data.
        """
        
        tree = XGBoostTree(loss_func=self._loss_func,
                                      pert_radius=self._pert_radius,
                                      max_depth=self._max_depth,
                                      min_samples_leaf=self._min_samples_leaf,
                                      lamda=self.lamda,
                                      gamma=self.gamma,
                                      learning_rate=self.learning_rate,
                                      rob_alpha=self.rob_alpha
                                      )
        
        return tree
        
