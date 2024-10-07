import pandas as pd
import os
from src.loanDefault import logger
from xgboost import XGBClassifier
import joblib
from src.loanDefault.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_x = pd.read_csv(self.config.x_train_data_path)
        train_y = pd.read_csv(self.config.y_train_data_path)
        test_x = pd.read_csv(self.config.x_test_data_path)
        test_y = pd.read_csv(self.config.y_test_data_path)

        lr = XGBClassifier(
                learning_rate=self.config.learning_rate,
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_child_weight=self.config.min_child_weight,
                gamma=self.config.gamma,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                objective=self.config.objective,
                nthread=self.config.nthread,
                scale_pos_weight=self.config.scale_pos_weight,
                seed=self.config.seed
            )
        
        lr.fit(train_x, train_y)

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))

