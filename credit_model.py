from pathlib import Path

import feast
import os
import numpy as np
from typing import Optional, Tuple
import xgboost as xgb
import mlflow
import mlflow.pyfunc
import joblib
import pandas as pd
from sklearn import tree
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
import ray
from ray.ml.predictors.integrations.xgboost import XGBoostPredictor
from ray.ml.train.integrations.xgboost import XGBoostTrainer
from ray.data.dataset import Dataset
from ray.ml.result import Result
from ray.ml.checkpoint import Checkpoint
import ray.cloudpickle as cpickle
from ray.ml.preprocessor import Preprocessor
from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray.ml import RunConfig



from ray.ml.checkpoint import Checkpoint
from ray.ml.constants import MODEL_KEY, PREPROCESSOR_KEY, TRAIN_DATASET_KEY

def load_from_checkpoint(checkpoint: Checkpoint,) -> Tuple[xgb.Booster, Optional[Preprocessor]]:
        checkpoint_path = checkpoint.to_directory()
        xgb_model = xgb.Booster()
        xgb_model.load_model(os.path.join(checkpoint_path, MODEL_KEY))
        preprocessor_path = os.path.join(checkpoint_path, PREPROCESSOR_KEY)
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, "rb") as f:
                preprocessor = cpickle.load(f)
        else:
            preprocessor = None

        return xgb_model, preprocessor

class CreditScoringModel:
    categorical_features = [
        "person_home_ownership",
        "loan_intent",
        "city",
        "state",
        "location_type",
    ]

    feast_features = [
        "zipcode_features:city",
        "zipcode_features:state",
        "zipcode_features:location_type",
        "zipcode_features:tax_returns_filed",
        "zipcode_features:population",
        "zipcode_features:total_wages",
        "credit_history:credit_card_due",
        "credit_history:mortgage_due",
        "credit_history:student_loan_due",
        "credit_history:vehicle_loan_due",
        "credit_history:hard_pulls",
        "credit_history:missed_payments_2y",
        "credit_history:missed_payments_1y",
        "credit_history:missed_payments_6m",
        "credit_history:bankruptcies",
    ]

    target = "loan_status"
    model_filename = "model.pkl"
    encoder_filename = "encoder.bin"


    def __init__(self):
        # Load model

        if Path(self.model_filename).exists():
            print("Loading saved model")
            self.checkpoint = ray.ml.checkpoint.Checkpoint(self.model_filename)

         # Load ordinal encoder
        if Path(self.encoder_filename).exists():
            self.encoder = joblib.load(self.encoder_filename)
        else:
            self.encoder = OrdinalEncoder()

        # Set up feature store
        self.fs = feast.FeatureStore(repo_path="feature_repo")

    def train(self, loans):
        train_X, train_Y = self._get_training_features(loans)

        dataset_df = pd.DataFrame(train_X)
        dataset_df["target"] = train_Y
        train_df, test_df = train_test_split(dataset_df, test_size=0.3)
        train_dataset = ray.data.from_pandas(train_df)
        valid_dataset = ray.data.from_pandas(test_df)
        test_dataset = ray.data.from_pandas(test_df.drop("target", axis=1))

        # XGBoost specific params
        params = {
            "tree_method": "approx",
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "error"],
        }

        trainer = XGBoostTrainer(
            scaling_config={
                "num_workers": 2,
                "use_gpu": 0,
            },
            label_column="target",
            params=params,
            datasets={"train": train_dataset, "valid": valid_dataset},
            #preprocessor=preprocessor,
            num_boost_round=100,
            run_config=RunConfig(callbacks=[MLflowLoggerCallback(experiment_name="loan-application", save_artifact=True)])
        )
        self.checkpoint = trainer.fit().checkpoint
        self.checkpoint.to_directory(self.model_filename)

    def _get_training_features(self, loans):
        training_df = self.fs.get_historical_features(
            entity_df=loans, features=self.feast_features
        ).to_df()
        
        self._fit_ordinal_encoder(training_df)
        self._apply_ordinal_encoding(training_df)

        train_X = training_df[
            training_df.columns.drop(self.target)
            .drop("event_timestamp")
            .drop("created_timestamp__")
            .drop("loan_id")
            .drop("zipcode")
            .drop("dob_ssn")
        ]
        train_X = train_X.reindex(sorted(train_X.columns), axis=1)
        train_Y = training_df.loc[:, self.target]

        return train_X, train_Y

    def _fit_ordinal_encoder(self, requests):
        self.encoder.fit(requests[self.categorical_features])
        joblib.dump(self.encoder, self.encoder_filename)

    def _apply_ordinal_encoding(self, requests):
        requests[self.categorical_features] = self.encoder.transform(
            requests[self.categorical_features]
        )

    #Todo Ray AIR predict for single payload.
    def predict(self, request):
        # Get online features from Feast
        feature_vector = self._get_online_features_from_feast(request)

        # Join features to request features
        features = request.copy()
        features.update(feature_vector)
        features_df = pd.DataFrame.from_dict(features)

        # Apply ordinal encoding to categorical features
        self._apply_ordinal_encoding(features_df)

        # Sort columns
        features_df = features_df.reindex(sorted(features_df.columns), axis=1)

        # Drop unnecessary columns
        features_df = features_df[features_df.columns.drop("zipcode").drop("dob_ssn")]
        print(f"Inference request: {features_df}")
        
        # Make prediction
        xgb_model, _ = load_from_checkpoint(self.checkpoint)
        
        features_df = np.array(features_df)
        prediction = xgb_model.inplace_predict(features_df)

        # return result of credit scoring
        return np.round(prediction)

    def _get_online_features_from_feast(self, request):
        zipcode = request["zipcode"][0]
        dob_ssn = request["dob_ssn"][0]

        return self.fs.get_online_features(
            entity_rows=[{"zipcode": zipcode, "dob_ssn": dob_ssn}],
            features=self.feast_features,
        ).to_dict()

    def is_model_trained(self):
        try:
            #self.is_model_trained()
            Path(self.model_filename).exists()
            #check_is_fitted(self.classifier, "tree_")
        except NotFittedError:
            return False
        return Path(self.model_filename).exists() #True
