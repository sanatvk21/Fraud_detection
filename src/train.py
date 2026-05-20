from xgboost import XGBClassifier
import joblib
import os
import mlflow
import mlflow.xgboost

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score
)

from preprocessing import (
    load_data,
    preprocess_data
)

# MLflow Configuration


mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "fraud_detection"
)


# Load Data

df = load_data(
    "data/creditcard.csv"
)

X_train, X_test, y_train, y_test, scaler = preprocess_data(
    df
)


# Handle Imbalanced Data


scale_pos_weight = (
    (y_train == 0).sum()
    /
    (y_train == 1).sum()
)

print(
    "Scale pos weight:",
    scale_pos_weight
)


# Start MLflow Run


with mlflow.start_run():

    # Parameters
    n_estimators = 200
    max_depth = 6
    learning_rate = 0.1


    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "max_depth",
        max_depth
    )

    mlflow.log_param(
        "learning_rate",
        learning_rate
    )


    # Create Model
    model = XGBClassifier(

        n_estimators=n_estimators,

        max_depth=max_depth,

        learning_rate=learning_rate,

        scale_pos_weight=scale_pos_weight,

        random_state=42,

        eval_metric="logloss"
    )


    # Train
    model.fit(
        X_train,
        y_train
    )


    print(
        "Training complete"
    )


    # Predictions
   

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:,1]


    # Metrics
   

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )


    print(
        "ROC AUC:",
        roc_auc
    )

    print(
        "Precision:",
        precision
    )

    print(
        "Recall:",
        recall
    )

    # Log Metrics
  

    mlflow.log_metric(
        "roc_auc",
        roc_auc
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )


    # Save Model
   

    os.makedirs(
        "models",
        exist_ok=True
    )


    joblib.dump(
        model,
        "models/xgb_model.pkl"
    )


    print(
        "Model saved"
    )

    # Log Model in MLflow
    

    mlflow.xgboost.log_model(
        xgb_model=model,
        name="model"
    )