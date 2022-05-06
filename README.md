# Reference architecture with best of breed OSS ML tools running on top of Ray AIR. 
# Feast & Ray 

This reference architecture contains an end to end example illustrating the following components.
- Ray AIR for scalable AI Runtime. (Data preprocessing, Train,Tune and batch)
- Feast for feature store
- Ray Serve for scalable, composable and framework agnostic ML model serving compute


## Overview

This tutorial demonstrates the use of best of breed ML tools as part of a real-time credit scoring application. It uses the feast on AWS example as a starting point.
* The primary training dataset is a loan table. This table contains historic loan data with accompanying features. The dataset also contains a target variable, namely whether a user has defaulted on their loan.
* We will be using the `run.py` to fetch the features from feast and train a scalable xgboost model using Ray. We do a simple test to make sure our model works as intended.
* We will then use a notebook to deploy a multi-step Ray Serve model endpoint to demonstrate how we can integrate retrieving online features using feast and decoupling those tasks for granular resources allocations and scaling.

The whole end to end reference implementation can be run on your laptop or in a collab notebook.



## Setup
### Setting up the environment
`conda create -n ray-demo python==3.8 pip`
<br>
`conda activate ray-demo`
<br>
`pip install -r requirements.txt`
<br>
### Setting up Feast locally


### Setting up Feast

We have already set up a feature repository in [feature_repo/](feature_repo/). It isn't necessary to create a new
feature repository, but it can be done using the following command
```
feast init -t local feature_repo # Command only shown for reference.
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

## Training our credit card model example.

Finally, we run the `run.py` script to train the model using a combination of loan data from our offline feature store and our zipcode, and then we test online inference by reading those same features from the online store.

Note that there are two scripts providing two levels of abstractions.

`run.py` provides the highest level of abstraction and allow a ML practictioner to iterate quickly without having to know the underlying tools, framework and infrastructure.

`credit_model.py` includes the CreditScoringModel() class and all the functions integrated with ray and feast.

Let's run our python run.py
```
The script should then output the result of a single loan application
```
loan rejected!
```
We are now ready to create our Ray Serve services using the jupyter notebook.

Open the notebook and follow the instructions.