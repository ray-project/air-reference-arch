import ray
from ray import serve
import requests
import numpy as np
from typing import List
import mlflow

import pydantic
from fastapi import FastAPI
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

import feast

from ray.ml.constants import MODEL_KEY as RAY_MODEL_KEY
from ray.ml.checkpoint import Checkpoint as RayCheckpoint

MODEL_FILENAME = "model.pkl"
ENCODER_FILENAME = "encoder.bin"

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

@serve.deployment
class CreditScoringPreprocessorDeployment:
    """The preprocessing part of CreditScoringModel.
    
    1. initialize the encoder and feature store varaible
    2. copy over `preprocess`, `apply_ordinal` and `get_online_feature` methods.
    3. implement __call__ to handle starlette request.
    """

    def __init__(self):
        self.encoder = joblib.load(ENCODER_FILENAME)
        self.fs = feast.FeatureStore(repo_path="feature_repo")
    
    def preprocess(self, model_request):
        # Get online features from Feast
        feature_vector = self._get_online_features_from_feast(model_request)

        # Join features to request features
        features = model_request.copy()
        features.update(feature_vector)
        features_df = pd.DataFrame.from_dict(features)

        # Apply ordinal encoding to categorical features
        self._apply_ordinal_encoding(features_df)

        # Sort columns
        features_df = features_df.reindex(sorted(features_df.columns), axis=1)

        # Drop unnecessary columns
        features_df = features_df[features_df.columns.drop("zipcode").drop("dob_ssn")]
        print("================================")
        print(f"Inference request:")
        print("--------------------------------")
        print(features_df.T)
        print("================================\n")
        return np.array(features_df)
    
    def _apply_ordinal_encoding(self, requests):
        requests[categorical_features] = self.encoder.transform(
            requests[categorical_features]
        )
    
    def _get_online_features_from_feast(self, request):
        zipcode = request["zipcode"][0]
        dob_ssn = request["dob_ssn"][0]

        return self.fs.get_online_features(
            entity_rows=[{"zipcode": zipcode, "dob_ssn": dob_ssn}],
            features=feast_features,
        ).to_dict()

    async def __call__(self, http_request):
        model_request = await http_request.json()
        features_array = self.preprocess(model_request)
        model_handle = serve.get_deployment("CreditScoringInferenceDeployment").get_handle()
        return ray.get(model_handle.predict.remote(features_array))

model_uri = "file:///home/ec2-user/air-reference-arch/mlruns/1/6e667136e621458bbbc580d9e25b3b48/artifacts/checkpoint_000099/"

@serve.deployment
class CreditScoringInferenceDeployment:
    """The inference part of CreditScoringModel.

    1. copy over load_from_checkpoint and predict method
    2. initialize the model
    3. change predict to drop the preprocess call
    """
    def __init__(self):
        self.model = self._load_from_checkpoint(RayCheckpoint(MODEL_FILENAME))
        #self.model = mlflow.pyfunc.load_model(model_uri=model_uri)
        
    def _load_from_checkpoint(self, checkpoint: RayCheckpoint):
        checkpoint_path = checkpoint.to_directory()
        xgb_model = xgb.Booster()
        xgb_model.load_model(os.path.join(checkpoint_path, RAY_MODEL_KEY))
        return xgb_model

    def predict(self, model_request):
        # Make prediction
        prediction = self.model.inplace_predict(model_request)

        # Return result of credit scoring
        return np.round(prediction)



ray.init()
serve.start(http_options={"host": "0.0.0.0"})
CreditScoringInferenceDeployment.deploy()
CreditScoringPreprocessorDeployment.deploy()

loan_request = {
    "zipcode": [76104],
    "dob_ssn": ["19630621_4278"],
    "person_age": [133],
    "person_income": [59000],
    "person_home_ownership": ["RENT"],
    "person_emp_length": [123.0],
    "loan_intent": ["PERSONAL"],
    "loan_amnt": [35000],
    "loan_int_rate": [16.02],
}

response = requests.post("http://localhost:8000/CreditScoringPreprocessorDeployment", json=loan_request)
result = response.json()[0]
print(f"Inference result: {result}\n")

if result == 0:
    print("Loan approved!\n")
elif result == 1:
    print("Loan rejected!\n")