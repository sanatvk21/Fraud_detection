import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Function to load dataset
def load_data(path):

    df = pd.read_csv(path)

    return df


# Function for preprocessing
def preprocess_data(df):

    # Separate features and target
    X = df.drop("Class", axis=1)

    y = df["Class"]


    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )


    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)


    return (

        X_train_scaled,

        X_test_scaled,

        y_train,

        y_test,

        scaler
    )