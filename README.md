# Reference architecture with best of breed OSS ML tools running on top of Ray AIR. 
# Feast Airflow Ray MlFlow Streamlit (FARMS)

This reference architecture contains an end to end example illustrating the following components.
- Ray AIR for scalable AI Runtime. (Data preprocessing, Train,Tune and batch)
- Feast for feature store
- MlFlow for experiment tracking and model registry
- Ray Serve for scalable, composable and framework agnostic ML model serving compute
- Airflow for workflow orchestration
- Streamlit for data application.

## Overview

This tutorial demonstrates the use of best of breed ML tools as part of a real-time credit scoring application. It uses the feast on AWS example as a starting point.
* The primary training dataset is a loan table. This table contains historic loan data with accompanying features. The dataset also contains a target variable, namely whether a user has defaulted on their loan.
* Feast and Ray is used during training to enrich the loan table with zipcode and credit history features from a S3 files.
* Feast is also used to serve the latest zipcode and credit history features for online credit scoring.

The whole end to end reference implementation can be run on your laptop or on a single host.



## Requirements

* feast
* ray

## Setup

### Setting up Feast locally


### Setting up Feast

Install Feast using pip

```
pip install feast, ray==1.13
```

We have already set up a feature repository in [feature_repo/](feature_repo/). It isn't necessary to create a new
feature repository, but it can be done using the following command
```
feast init -t aws feature_repo # Command only shown for reference.
```

Since we don't need to `init` a new repository, all we have to do is configure the 
[feature_store.yaml/](feature_repo/feature_store.yaml) in the feature repository. This file has been configured to be running feast locally.

Deploy the feature store by running `apply` from within the `feature_repo/` folder
```
cd feature_repo/
feast apply
```
```
Registered entity dob_ssn
Registered entity zipcode
Registered feature view credit_history
Registered feature view zipcode_features
Deploying infrastructure for credit_history
Deploying infrastructure for zipcode_features
```

Next we load features into the online store using the `materialize-incremental` command. This command will load the
latest feature values from a data source into the online store.

```
CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
feast materialize-incremental $CURRENT_TIME
```

Return to the root of the repository
```
cd ..
```

## Running our end to end example credit card model example.

Finally, we train the model using a combination of loan data from S3 and our zipcode and credit history features from our offline store, and then we test online inference by reading those same features from the online store.

Note that there are two levels of abstractions.

`run.py` provides the highest level of abstraction and allow a ML practictioner to iterate quickly without having to know the tools, framework and infrastructure.

`credit_model.py` provides the CreditScoringModel() class and all the functions integrated with the FARMS framework.
```
# Create model
myModel = CreditScoringModel()

# Get historic loan data
loans = pd.read_parquet("data/loan_table.parquet")

result = myModel.predict(loan_request)
```
Let's run python run.py
```
The script should then output the result of a single loan application
```
loan rejected!
```
