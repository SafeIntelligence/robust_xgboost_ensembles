"""
Primary script to run the experiments reported in the paper. We compare our
proposed approach to the baselines on the benchmark suite from the paper
'Why do tree-based models still outperform deep learning on 
tabular data?' <https://arxiv.org/pdf/2207.08815>. 
"""

# Standard library imports
import argparse
import itertools
import json
import os
import pickle
import random
import time
from datetime import datetime
from pathlib import Path

# Third-party imports
import numpy as np
import openml
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Local imports
from robust_xgboost.baseline_interfaces.rgbdt_interface import RGBDTRegressor
from robust_xgboost.baseline_interfaces.vanilla_xgboost_interface import VanillaXGBoostRegressor
from robust_xgboost.evaluation.milp_solver_regression import MILPSolverRegression
from robust_xgboost.models.ensembles.xgboost_regressor import XGBoostRegressor

def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.

    Args:
        seed (int): The random seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)


def infer_xgb_ensemble(model_dict_list, x, base_score):
    """
    Conducts inference on an XGBoost ensemble by aggregating predictions from all trees.
    
    Args:
        model_dict_list (list): List of tree dictionaries representing the ensemble.
        x (array-like): Input features for prediction.
        base_score (float): Base prediction score.
    
    Returns:
        float: Final ensemble prediction.
    """
    pred = base_score
    
    for model_dict in model_dict_list:
        out = infer_xgb_tree(model_dict, x)
        pred += out
    
    return pred


def infer_xgb_tree(model_dict, x):
    """
    Conducts inference on a single XGBoost tree.
    
    Args:
        model_dict (dict): Tree structure dictionary.
        x (array-like): Input features for prediction.
    
    Returns:
        float: Tree prediction value.
    """
    # If this is a leaf node, return the leaf value
    if "leaf" in model_dict:
        return model_dict["leaf"]
    
    # Internal node - need to traverse further
    if isinstance(model_dict["split"], str):
        # Handle string feature names (e.g., "f0" -> 0)
        split_feat = int(model_dict["split"][1:])
    else:
        split_feat = model_dict["split"]
        
    split_thresh = model_dict["split_condition"]
    
    # Determine which child to traverse
    if x[split_feat] < split_thresh:
        child_nodeid = model_dict["yes"]
    else:
        child_nodeid = model_dict["no"]
        
    # Find the child node dictionary
    child_node_dict = None
    for cd in model_dict["children"]:
        if cd["nodeid"] == child_nodeid:
            child_node_dict = cd
            break
    
    # Recursively traverse the child node
    return infer_xgb_tree(child_node_dict, x)
    
def train_model(
    X_train,
    y_train,
    model_type,
    model_save_filepath,
    eps,
    base_score,
    n_base_models=30,
    max_depth=4,
    learning_rate=0.2,
    lamda=0.2,
    gamma=0.1,
    min_samples_leaf=5,
    alpha=1.0,
):
    """
    Trains a model based on the specified type and hyperparameters.
    
    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training targets.
        model_type (str): Type of model to train ("ours", "rgbdt", "vanilla_xgboost").
        model_save_filepath (str): Path to save the trained model.
        eps (float): Perturbation radius for robust training.
        base_score (float): Base prediction score.
        n_base_models (int): Number of base models/trees. Defaults to 30.
        max_depth (int): Maximum depth of trees. Defaults to 4.
        learning_rate (float): Learning rate. Defaults to 0.2.
        lamda (float): L2 regularization parameter. Defaults to 0.2.
        gamma (float): Minimum loss reduction for splits. Defaults to 0.1.
        min_samples_leaf (int): Minimum samples per leaf. Defaults to 5.
        alpha (float): Robustness parameter. Defaults to 1.0.
    
    Returns:
        float: Training time in seconds.
    """
    if model_type == "ours":
        ensemble = XGBoostRegressor(
            n_base_models=n_base_models,
            max_depth=max_depth,
            learning_rate=learning_rate,
            lamda=lamda,
            gamma=gamma,
            min_samples_leaf=min_samples_leaf,
            pert_radius=eps,
            rob_alpha=alpha,
            initial_pred=base_score
        )
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)

    elif model_type.lower() == "rgbdt":
        ensemble = RGBDTRegressor(
            num_trees=n_base_models,
            max_depth=max_depth,
            pert_radius=eps,
            learning_rate=learning_rate,
            base_score=base_score,
            gamma=gamma,
            lamda=lamda,
            task="regression",
            min_samples_leaf=min_samples_leaf,
        )

    elif model_type.lower() == "vanilla_xgboost":
        ensemble = VanillaXGBoostRegressor(
            objective='reg:squarederror',
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_base_models,
            subsample=1,
            colsample_bytree=1,
            reg_lambda=lamda,
            reg_alpha=0.0,
            gamma=gamma,
            random_state=42,
            base_score=base_score,
            tree_method='exact',
            min_child_weight=1,
        )

    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported")

    print(f"Training model {model_type}, with num_trees = {n_base_models}, "
          f"eps = {eps}. Saving at {model_save_filepath}")

    start = time.time()
    ensemble.fit(X_train, y_train)
    time_taken = time.time() - start

    ensemble.save_model(model_save_filepath)

    return time_taken

def verify_model(X_test, y_test, eps, model_path, base_score, output_pert_range):
    """
    Verifies the robustness of a trained model against adversarial perturbations.
    
    Args:
        X_test (array-like): Test features.
        y_test (array-like): Test targets.
        eps (float): Perturbation radius for verification.
        model_path (str): Path to the saved model.
        base_score (float): Base prediction score.
        output_pert_range (list): Range of output perturbation multipliers.
    
    Returns:
        dict: Dictionary containing verification results and metrics.
    """
    model_path = str(model_path)
    
    # Load the trained model
    with open(model_path, "r") as f:
        json_model = json.load(f)
    
    # Initialize result containers
    preds = []
    max_deviation_from_pred = []
    max_deviation_from_labels = []
    worst_case_predictions = []
    
    # Calculate dataset statistics
    std = np.std(y_test)
    mean = base_score
    
    # Initialize MILP solver for robustness verification
    solver = MILPSolverRegression(json_model=json_model, epsilon=eps)

    # Verify each test sample
    for i in tqdm(range(len(X_test))):
        x, y = X_test[i], y_test[i]
        
        # Get normal prediction
        pred = infer_xgb_ensemble(json_model, x, base_score)
        preds.append(pred)
        
        # Get worst-case prediction bounds using MILP solver
        min_val, max_val = solver.minimum_maximum_predictions(sample=x)
        
        # Add base score to solver outputs
        min_val += base_score
        max_val += base_score
        
        # Find worst-case prediction compared to true label
        worst_case_pred_compared_to_label = (
            min_val if abs(min_val - y) > abs(max_val - y) else max_val
        )
        worst_case_predictions.append(worst_case_pred_compared_to_label)
        max_deviation_from_labels.append(abs(worst_case_pred_compared_to_label - y))
        
        # Find worst-case prediction compared to normal prediction
        worst_case_pred_compared_to_pred = (
            min_val if abs(min_val - pred) > abs(max_val - pred) else max_val
        )
        max_deviation_from_pred.append(abs(worst_case_pred_compared_to_pred - pred))

    # Calculate performance metrics
    number_points = len(y_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    range_labels = np.max(y_test) - np.min(y_test)

    # Calculate worst-case metrics
    worst_case_mse = mean_squared_error(y_test, worst_case_predictions)
    worst_case_mae = np.mean(max_deviation_from_labels)
    worst_case_r2 = r2_score(y_test, worst_case_predictions)
    average_deviation_from_preds = np.mean(max_deviation_from_pred)

    # Compile base results
    results = {
        "verification_eps": eps,
        "mse": mse,
        "r2": r2,
        "worst_case_mse": worst_case_mse,
        "worst_case_mae": worst_case_mae,
        "worst_case_r2": worst_case_r2,
        "average_deviation_from_preds": average_deviation_from_preds,
        "std": std,
        "mean": mean,
        "range_labels": range_labels
    }

    print(f"Computing robust accuracy of {model_path} w.r.t radius {eps}")

    # Calculate robustness percentages for different output perturbation ranges
    for output_perturbation_multiplier in output_pert_range:
        # Robustness w.r.t. predictions
        num_unrobust_points_preds = np.sum(
            max_deviation_from_pred > range_labels * output_perturbation_multiplier
        )
        rob_acc_preds = (1 - num_unrobust_points_preds / number_points) * 100

        # Robustness w.r.t. labels
        num_unrobust_points_labels = np.sum(
            max_deviation_from_labels > range_labels * output_perturbation_multiplier
        )
        rob_acc_labels = (1 - num_unrobust_points_labels / number_points) * 100

        # Store results
        results[f"rob_acc_preds_range_{output_perturbation_multiplier}"] = rob_acc_preds
        results[f"rob_acc_labels_range_{output_perturbation_multiplier}"] = rob_acc_labels

    results["total"] = number_points

    return results
    
def run_experiments_kfold(
    dataset,
    X,
    y,
    model_type,
    train_eps,
    verification_eps_list,
    output_pert_ranges,
    models_dataset_dir,
    random_state=42,
    max_train_examples=8000,
    max_test_examples=1000,
    **kwargs
):
    """
    Runs k-fold cross-validation experiments for a given model type and dataset.
    
    Args:
        dataset (str): Name of the dataset.
        X (array-like): Input features.
        y (array-like): Target values.
        model_type (str): Type of model to train.
        train_eps (float): Training perturbation radius.
        verification_eps_list (list): List of verification perturbation radii.
        output_pert_ranges (list): List of output perturbation ranges.
        models_dataset_dir (Path): Directory to save models.
        random_state (int): Random state for reproducibility. Defaults to 42.
        max_train_examples (int): Maximum training examples. Defaults to 8000.
        max_test_examples (int): Maximum test examples. Defaults to 1000.
        **kwargs: Additional hyperparameters.
    
    Returns:
        DataFrame: Aggregated experimental results.
    """
    # Normalize features
    X = MinMaxScaler().fit_transform(X)
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Extract hyperparameters from kwargs
    alpha = kwargs.get('alpha', 1.0)
    worst_case_predictions = kwargs.get('worst_case_predictions', False)
    n_base_models = kwargs.get('n_base_models', 100)
    max_depth = kwargs.get('max_depth', 10)
    learning_rate = kwargs.get('learning_rate', 0.2)
    lamda = kwargs.get('lamda', 0.2)
    gamma = kwargs.get('gamma', 0.1)
    min_samples_leaf = kwargs.get('min_samples_leaf', 5)

    verification_results_list = []
    fold = 0

    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Limit dataset sizes for computational efficiency
        X_train = X_train[:max_train_examples]
        y_train = y_train[:max_train_examples]

        # Decrease test set size as verification can be slow
        X_test = X_test[:max_test_examples]
        y_test = y_test[:max_test_examples]

        mean_y = np.mean(y_train)

        # Define model save path
        model_save_filepath = str(
            models_dataset_dir / 
            f"{model_type}_{n_base_models}_eps_{train_eps}_alpha_{alpha}_"
            f"worst_case_{worst_case_predictions}_fold_{fold}.json"
        )

        # Train the model
        time_taken = train_model(
            X_train=X_train,
            y_train=y_train,
            model_type=model_type,
            model_save_filepath=model_save_filepath,
            eps=train_eps,
            base_score=mean_y,
            n_base_models=n_base_models,
            max_depth=max_depth,
            learning_rate=learning_rate,
            lamda=lamda,
            gamma=gamma,
            min_samples_leaf=min_samples_leaf
        )

        # Create verification results directory
        verification_result_dir = models_dataset_dir / "verification_results"
        if not os.path.exists(verification_result_dir):
            os.makedirs(verification_result_dir)

        # Verify model for each verification epsilon
        for verification_eps in verification_eps_list:
            results_verification_eps = {
                "train_eps": train_eps,
                "worst_case_training": int(worst_case_predictions),
                "time_taken": time_taken,
            }

            # Run verification
            verification_results = verify_model(
                X_test=X_test,
                y_test=y_test,
                eps=verification_eps,
                model_path=model_save_filepath,
                base_score=mean_y,
                output_pert_range=output_pert_ranges
            )

            # Save verification results
            verification_results_fp = (
                verification_result_dir / 
                f"{model_type}_train_eps_{train_eps}_test_eps_{verification_eps}_"
                f"alpha_{alpha}_worst_case_{worst_case_predictions}_fold_{fold}.json"
            )

            # Update results with verification metrics and hyperparameters
            results_verification_eps.update(verification_results)
            results_verification_eps.update({
                "num_trees": n_base_models,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "lamda": lamda,
                "gamma": gamma,
                "min_samples_leaf": min_samples_leaf,
                "alpha": alpha
            })

            # Save results to file
            with open(verification_results_fp, "w") as f:
                json.dump(results_verification_eps, f)

            verification_results_list.append(results_verification_eps)

        fold += 1

    # Aggregate results across folds
    results_df = pd.DataFrame(verification_results_list)
    numeric_cols = results_df.select_dtypes(include=['int64', 'float64']).columns

    # Group and calculate both mean and std for numeric columns only
    agg_stats = results_df.groupby('verification_eps')[numeric_cols].agg(['mean', 'std']).round(3)

    # Flatten column names - from multi-index to single index
    agg_stats.columns = ['_'.join(col).strip() for col in agg_stats.columns.values]

    # Reset index to make verification_eps a regular column
    agg_stats = agg_stats.reset_index()

    mean_cols = [col for col in agg_stats.columns if col.endswith('_mean')]

    # For each mean column, create a combined column with ± notation
    for mean_col in mean_cols:
        # Get corresponding std column name
        std_col = mean_col.replace('_mean', '_std')
        # Create new column name without _mean suffix
        new_col = mean_col.replace('_mean', '')

        # Combine mean and std, only add ± std if std is not 0
        agg_stats[new_col] = agg_stats.apply(
            lambda row: f"{row[mean_col]}" if row[std_col] == 0
            else f"{row[mean_col]} ± {row[std_col]}",
            axis=1
        )

        # Drop original mean and std columns
        agg_stats = agg_stats.drop([mean_col, std_col], axis=1)

    # Add dataset and model type columns as the first two columns
    agg_stats.insert(0, 'dataset', [dataset] * len(agg_stats))
    agg_stats.insert(1, 'model_type', [model_type] * len(agg_stats))

    return agg_stats
    
def run_experiments_across_train_settings(
    dataset, X, y, eps, results_filepath, models_dataset_dir, running_agg_df
):
    """
    Runs experiments across different training settings with hyperparameter grid search.
    
    Args:
        dataset (str): Name of the dataset.
        X (array-like): Input features.
        y (array-like): Target values.
        eps (float): Perturbation radius.
        results_filepath (Path): Path to save results.
        models_dataset_dir (Path): Directory to save models.
        running_agg_df (DataFrame): Running aggregated results dataframe.
    
    Returns:
        DataFrame: Updated aggregated results dataframe.
    """
    # Set smaller sample sizes for efficiency
    max_train_samples = 5000
    max_test_samples = 500

    # Load hyperparameters for the dataset
    hyperparams_fp = Path(__file__).resolve().parent / "hyperparams.json"
    with open(hyperparams_fp, "r") as f:
        hyperparams = json.load(f)[dataset]

    # Define experimental settings
    methods_list = [
        "ours",
        # "rgbdt",  # Commented out for current experiments
        "vanilla_xgboost"
    ]
    train_eps_list = [float(eps)]
    output_pert_ranges = [0.2, 0.4, 0.6, 0.8, 1]
    verification_eps_list = [float(eps)]

    # Grid search hyperparameters
    num_trees = [int(hyperparams["n_estimators"] / 10)]

    # Manually perform a grid sweep from the initial hyperparameters
    learning_rate = [
        float(hyperparams["learning_rate"]) * 0.5,
        float(hyperparams["learning_rate"]),
        float(hyperparams["learning_rate"]) * 2
    ]
    max_depth = [
        max(1, int(hyperparams["max_depth"]) - 2),
        int(hyperparams["max_depth"]),
        int(hyperparams["max_depth"]) + 2
    ]
    lamda = [
        float(hyperparams["reg_lambda"]) * 0.5,
        float(hyperparams["reg_lambda"]),
        float(hyperparams["reg_lambda"]) * 2
    ]

    # Fixed hyperparameters
    rob_alpha = [1]
    gamma = [float(hyperparams["gamma"])]
    min_samples_leaf = [int(hyperparams["min_child_weight"])]

    # Generate all parameter combinations
    param_combinations = itertools.product(
        methods_list,
        train_eps_list,
        num_trees,
        max_depth,
        learning_rate,
        rob_alpha,
        lamda,
        gamma,
        min_samples_leaf
    )

    # Run experiments for each parameter combination
    for (model_type, train_eps, n_trees, depth, lr, ralpha, lam, gam, min_leaf) in param_combinations:
        agg_stats = run_experiments_kfold(
            dataset=dataset,
            X=X,
            y=y,
            model_type=model_type,
            train_eps=train_eps,
            verification_eps_list=verification_eps_list,
            output_pert_ranges=output_pert_ranges,
            models_dataset_dir=models_dataset_dir,
            max_train_examples=max_train_samples,
            max_test_examples=max_test_samples,
            alpha=ralpha,
            n_base_models=n_trees,
            max_depth=depth,
            learning_rate=lr,
            lamda=lam,
            gamma=gam,
            min_samples_leaf=min_leaf
        )

        # Aggregate results and save incrementally
        running_agg_df = pd.concat([running_agg_df, agg_stats], axis=0)
        running_agg_df.to_csv(results_filepath, index=False)

    return running_agg_df

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run robust XGBoost regression experiments on OpenML benchmark suite"
    )
    parser.add_argument(
        "--eps", 
        type=float, 
        default=0.05,
        help="Perturbation radius for robust training and verification (default: 0.05)"
    )

    args = parser.parse_args()
    eps = args.eps

    print(f"Running experiments with eps = {eps}")

    # Set random seed for reproducibility
    set_seed(42)

    # Initialize results dataframe
    running_agg_df = pd.DataFrame()

    # Create unique identifier for this experiment run
    current_time = datetime.now().strftime("%d-%m-%Y-%H-%M")
    identifier = f"{current_time}_{eps}"

    # Setup results directory
    results_dir = Path(__file__).resolve().parent / "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_fp = results_dir / f"results_{identifier}.csv"

    # OpenML benchmark suite for regression on numerical features
    SUITE_ID = 336

    # Try to get the benchmark suite with retry mechanism for robustness
    max_retries = 10
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Check if we have a cached version first
            cache_path = (
                Path(__file__).resolve().parent / "cache" / f"benchmark_suite_{SUITE_ID}.pkl"
            )
            cache_path.parent.mkdir(exist_ok=True, parents=True)

            if cache_path.exists():
                print(f"Loading cached benchmark suite from {cache_path}")
                with open(cache_path, "rb") as f:
                    benchmark_suite = pickle.load(f)
            else:
                # If not cached, fetch from OpenML
                print(f"Fetching benchmark suite {SUITE_ID} from OpenML...")
                benchmark_suite = openml.study.get_suite(SUITE_ID)

                # Cache the result for future use
                with open(cache_path, "wb") as f:
                    pickle.dump(benchmark_suite, f)
                print(f"Cached benchmark suite to {cache_path}")
            
            break  # Exit the loop if successful
            
        except Exception as e:
            retry_count += 1
            print(f"Attempt {retry_count}/{max_retries} failed to fetch benchmark suite: {e}")
            
            if retry_count < max_retries:
                wait_time = 5  # Wait 5 seconds between retries
                print(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print(f"Failed to fetch benchmark suite after {max_retries} attempts.")
                raise Exception("Unable to fetch benchmark suite after multiple attempts.")

    # Iterate over all tasks in the benchmark suite
    for task_id in benchmark_suite.tasks:
        try:
            # Download the OpenML task and dataset
            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()
            dataset_name = dataset.name

            print(f"Running experiments on dataset {dataset_name}...")

            # Load data from OpenML
            data = fetch_openml(data_id=dataset.id)
            X = data.data.values
            y = data.target.values

            # Normalize features
            X = MinMaxScaler().fit_transform(X)

            # Create dataset-specific model directory
            models_dataset_dir = (
                Path(__file__).resolve().parent / "models" / f"{identifier}" / dataset_name
            )
            os.makedirs(models_dataset_dir)

            # Run experiments for this dataset
            running_agg_df = run_experiments_across_train_settings(
                dataset=dataset_name,
                X=X,
                y=y,
                eps=eps,
                results_filepath=results_fp,
                models_dataset_dir=models_dataset_dir,
                running_agg_df=running_agg_df,
            )

        except Exception as e:
            print(f"Error in dataset {dataset_name}: {e}")

            # Log errors to file
            errors_dir = Path(__file__).resolve().parent / "errors"
            if not os.path.exists(errors_dir):
                os.makedirs(errors_dir)

            with open(errors_dir / f"{identifier}_errors.txt", "a") as f:
                f.write(f"Error in dataset {dataset_name}: {e}\n")

            continue

    print(f"Experiments completed. Results saved to {results_fp}")