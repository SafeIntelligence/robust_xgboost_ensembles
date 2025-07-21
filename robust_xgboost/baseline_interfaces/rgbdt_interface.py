"""
Interface for using the RGBDT model from 
'Robust Decision Trees Against Adversarial Examples', Chen et al. (2019).

The RGBDT library is a C++ implementation, which must be built from the 
instructions listed in the original library (https://github.com/chenhongge/RobustTrees)
"""

import os
import uuid
from pathlib import Path
import time
import shutil
import subprocess

def numpy_to_chensvmlight(X, y, filename):
    """
    Export a numpy dataset to the SVM-Light format that is needed for Chen et al. (2019).

    The difference between SVM-Light and this format is that zero values are also included.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Array of amples.
    y : array-like of shape (n_samples,)
        Array of class labels as integers.
    filename : str
        Exported SVM-Light dataset filename or path.
    """
    lines = []
    for sample, label in zip(X, y):
        terms = [str(label)]
        for i, value in enumerate(sample):
            terms.append(f"{i}:{value}")
        lines.append(" ".join(terms))
    svmlight_string = "\n".join(lines)

    with open(filename, "w") as file:
        file.write(svmlight_string)
        
class RGBDTRegressor:
    
    def __init__(
        self, 
        num_trees,
        max_depth,
        pert_radius=0,
        learning_rate=0.2,
        base_score=0.0,
        gamma=0.1,
        lamda=0.2,
        task="regression",
        min_samples_leaf=1,
    ):
        
        self.parent_dir = str(Path(__file__).resolve().parent)
        
        # this library has a C++ interface so it will be called using subprocess and saved files, which will be later deleted
        self.intermediate_dir = f"{self.parent_dir}/robust_trees_intermediate_{int(time.time())}_{uuid.uuid4()}"
        
        os.mkdir(self.intermediate_dir)
        
        if task == "regression":
            objective = "reg:linear"
            
        elif task == "classification":
            objective = "binary:logistic"
            
        else:
            raise ValueError(f"Task {task} not recognized.")
        
        self.train_config = {
            "booster": "gbtree",
            "objective": objective,
            "tree_method": "robust_exact",
            "eta": learning_rate,
            "gamma": gamma,
            "min_child_weight": min_samples_leaf,
            "max_depth": max_depth,
            "num_round": num_trees,
            "save_period": 0,
            "nthread": 1,
            "data": f"{self.intermediate_dir}/data.train",
            "eval[test]": f"{self.intermediate_dir}/data.train",    # we are not using a validation set, therefore the train set is used as val set
            "test:data": f"{self.intermediate_dir}/data.test",
            "model_out": f"{self.intermediate_dir}/model.model",
            "robust_eps": pert_radius,
            "subsample": 1,
            "colsample_bytree": 1,
            "reg_lambda": lamda,
            "reg_alpha": 0.0,
            "base_score": base_score,
        }
        
        self.train_config_filepath = f"{self.intermediate_dir}/train_config.conf"
        
        with open(self.train_config_filepath,"w") as f:
            for key, value in self.train_config.items():
                f.write(f"{key} = {value}\n")

        self.dump_config = {
            "task": "dump",
            "dump_format": "json",
            "model_in": f"{self.intermediate_dir}/model.model",
            "model_out": f"{self.intermediate_dir}/model.json",
            "name_dump": f"{self.intermediate_dir}/model.json"
        }
        
        self.dump_config_filepath = f"{self.intermediate_dir}/dump_config.conf"
        
        with open(self.dump_config_filepath,"w") as f:
            for key, value in self.dump_config.items():
                f.write(f"{key} = {value}\n")
            
    def fit(
        self,
        train_data,
        train_labels,
    ):
        
        numpy_to_chensvmlight(train_data, train_labels, f"{self.intermediate_dir}/data.train")
        
        # train the models
        subprocess.run([f"{self.parent_dir}/xgboost", self.train_config_filepath])

        # save them as json
        subprocess.run([f"{self.parent_dir}/xgboost", self.dump_config_filepath])

    
    def save_model(
        self,
        filepath
    ):
        
        shutil.copy(f"{self.intermediate_dir}/model.json", filepath)
        
    def __del__(
        self,
    ):
        shutil.rmtree(self.intermediate_dir)