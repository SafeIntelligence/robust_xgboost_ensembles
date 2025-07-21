import unittest
import numpy as np
from robust_xgboost.models.decision_trees.losses import XGBoostLogisticLoss, XGBoostMSELoss

class TestXGBoostDerivatives(unittest.TestCase):
    
    def setUp(self):
        self.logistic_loss = XGBoostLogisticLoss()
        self.mse_loss = XGBoostMSELoss()
    
    # --- Logistic Loss First Derivative Tests ---
    
    def test_logistic_loss_first_derivative_simple(self):
        """Test simple cases for logistic loss first derivative."""
        annotations = np.array([0, 1, 0, 1])
        pred_prob_prev = np.array([0.2, 0.8, 0.3, 0.6])
        
        expected = -(annotations - pred_prob_prev)
        result = self.logistic_loss._calculate_first_derivative(annotations, pred_prob_prev)
        
        np.testing.assert_almost_equal(result, expected)
        self.assertEqual(result.shape, annotations.shape) # Shape test

    def test_logistic_loss_first_derivative_edge_cases(self):
        """Test edge cases for logistic loss first derivative."""
        # Edge case: all zeros
        annotations = np.array([0, 0, 0, 0])
        pred_prob_prev = np.array([0.2, 0.3, 0.4, 0.5])
        expected = -(annotations - pred_prob_prev)
        result = self.logistic_loss._calculate_first_derivative(annotations, pred_prob_prev)
        np.testing.assert_almost_equal(result, expected)
        self.assertEqual(result.shape, annotations.shape) # Shape test
        
        # Edge case: all ones
        annotations = np.array([1, 1, 1, 1])
        # Use the same pred_prob_prev as above for consistency
        expected = -(annotations - pred_prob_prev) 
        result = self.logistic_loss._calculate_first_derivative(annotations, pred_prob_prev)
        np.testing.assert_almost_equal(result, expected)
        self.assertEqual(result.shape, annotations.shape) # Shape test

    def test_logistic_loss_first_derivative_random(self):
        """Test random arrays for logistic loss first derivative."""
        np.random.seed(42)
        for _ in range(3):  # Test with 3 different random arrays
            size = np.random.randint(10, 100)
            annotations = np.random.randint(0, 2, size=size)
            pred_prob_prev = np.random.random(size=size)
            
            expected = -(annotations - pred_prob_prev)
            result = self.logistic_loss._calculate_first_derivative(annotations, pred_prob_prev)
            
            np.testing.assert_almost_equal(result, expected)
            self.assertEqual(result.shape, annotations.shape) # Shape test

    # --- Logistic Loss Second Derivative Tests ---

    def test_logistic_loss_second_derivative_simple(self):
        """Test simple cases for logistic loss second derivative."""
        annotations = np.array([0, 1, 0, 1]) # Annotations are not used but passed for consistency
        pred_prob_prev = np.array([0.2, 0.8, 0.3, 0.6])
        
        expected = pred_prob_prev * (1 - pred_prob_prev)
        result = self.logistic_loss._calculate_second_derivative(annotations, pred_prob_prev)
        
        np.testing.assert_almost_equal(result, expected)
        self.assertEqual(result.shape, pred_prob_prev.shape) # Shape test

    def test_logistic_loss_second_derivative_edge_cases(self):
        """Test edge cases for logistic loss second derivative."""
        annotations = np.array([0, 0, 0, 0]) # Not used
        # Edge case: probabilities near 0 and 1
        pred_prob_prev = np.array([0.01, 0.99, 0.5, 0.0]) 
        expected = pred_prob_prev * (1 - pred_prob_prev)
        result = self.logistic_loss._calculate_second_derivative(annotations, pred_prob_prev)
        np.testing.assert_almost_equal(result, expected)
        self.assertEqual(result.shape, pred_prob_prev.shape) # Shape test

    def test_logistic_loss_second_derivative_random(self):
        """Test random arrays for logistic loss second derivative."""
        np.random.seed(43) # Use different seed
        for _ in range(3):
            size = np.random.randint(10, 100)
            annotations = np.random.randint(0, 2, size=size) # Not used
            pred_prob_prev = np.random.random(size=size)
            
            expected = pred_prob_prev * (1 - pred_prob_prev)
            result = self.logistic_loss._calculate_second_derivative(annotations, pred_prob_prev)
            
            np.testing.assert_almost_equal(result, expected)
            self.assertEqual(result.shape, pred_prob_prev.shape) # Shape test

    # --- MSE Loss First Derivative Tests ---
    
    def test_mse_loss_first_derivative_simple(self):
        """Test simple cases for MSE loss first derivative."""
        annotations = np.array([0, 1, 2, 3])
        pred_prob_prev = np.array([0.5, 1.5, 1.8, 2.9])
        
        expected = -(annotations - pred_prob_prev)
        result = self.mse_loss._calculate_first_derivative(annotations, pred_prob_prev)
        
        np.testing.assert_almost_equal(result, expected)
        self.assertEqual(result.shape, annotations.shape) # Shape test

    def test_mse_loss_first_derivative_edge_cases(self):
        """Test edge cases for MSE loss first derivative."""
        # Edge case: identical arrays
        annotations = np.array([1.0, 2.0, 3.0, 4.0])
        pred_prob_prev = np.array([1.0, 2.0, 3.0, 4.0])
        expected = np.zeros_like(annotations)
        result = self.mse_loss._calculate_first_derivative(annotations, pred_prob_prev)
        np.testing.assert_almost_equal(result, expected)
        self.assertEqual(result.shape, annotations.shape) # Shape test
        
        # Edge case: negative values
        annotations = np.array([-1.0, -2.0, 3.0, 4.0])
        pred_prob_prev = np.array([1.0, 2.0, -3.0, -4.0])
        expected = -(annotations - pred_prob_prev)
        result = self.mse_loss._calculate_first_derivative(annotations, pred_prob_prev)
        np.testing.assert_almost_equal(result, expected)
        self.assertEqual(result.shape, annotations.shape) # Shape test

    def test_mse_loss_first_derivative_random(self):
        """Test random arrays for MSE loss first derivative."""
        np.random.seed(42)
        for _ in range(3):
            size = np.random.randint(10, 100)
            annotations = np.random.normal(size=size)
            pred_prob_prev = np.random.normal(size=size)
            
            expected = -(annotations - pred_prob_prev)
            result = self.mse_loss._calculate_first_derivative(annotations, pred_prob_prev)
            
            np.testing.assert_almost_equal(result, expected)
            self.assertEqual(result.shape, annotations.shape) # Shape test

    # --- MSE Loss Second Derivative Tests ---

    def test_mse_loss_second_derivative_simple(self):
        """Test simple cases for MSE loss second derivative."""
        annotations = np.array([0, 1, 2, 3]) # Not used but passed for consistency
        pred_prob_prev = np.array([0.5, 1.5, 1.8, 2.9]) # Not used
        
        expected = np.ones_like(annotations) # Second derivative is always 1
        result = self.mse_loss._calculate_second_derivative(annotations, pred_prob_prev)
        
        np.testing.assert_almost_equal(result, expected)
        self.assertEqual(result.shape, annotations.shape) # Shape test

    def test_mse_loss_second_derivative_random(self):
        """Test random arrays for MSE loss second derivative."""
        np.random.seed(44) # Use different seed
        for _ in range(3):
            size = np.random.randint(10, 100)
            annotations = np.random.normal(size=size) # Not used
            pred_prob_prev = np.random.normal(size=size) # Not used
            
            expected = np.ones_like(annotations) # Second derivative is always 1
            result = self.mse_loss._calculate_second_derivative(annotations, pred_prob_prev)
            
            np.testing.assert_almost_equal(result, expected)
            self.assertEqual(result.shape, annotations.shape) # Shape test


if __name__ == '__main__':
    unittest.main()