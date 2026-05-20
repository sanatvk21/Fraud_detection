import joblib

from sklearn.metrics import (

    classification_report,

    roc_auc_score,

    confusion_matrix
)

from preprocessing import (
    load_data,
    preprocess_data
)


# Load data
df = load_data("data/creditcard.csv")

X_train, X_test, y_train, y_test, scaler = preprocess_data(df)


# Load saved model
model = joblib.load("models/xgb_model.pkl")


# Predictions
y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]


# Classification report
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ROC AUC
roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("ROC AUC:", roc_auc)


# Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred
)

print("Confusion Matrix:\n", cm)