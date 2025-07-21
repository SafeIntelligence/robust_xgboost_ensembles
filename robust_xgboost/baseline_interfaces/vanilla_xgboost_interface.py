"""
Interface for using the conventional XGBoost regressor
"""

from xgboost import XGBRegressor

class VanillaXGBoostRegressor(XGBRegressor):
    
    def save_model(
        self,
        filepath
    ):
        booster = self.get_booster()
        booster.dump_model(filepath, dump_format='json')