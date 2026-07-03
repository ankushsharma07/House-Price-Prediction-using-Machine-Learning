import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    """Load the house price dataset."""
    return pd.read_csv("data/house_prices.csv")


def handle_missing_values(df):
    """Handle missing values."""

    print("Missing Values Before Cleaning:")
    print(df.isnull().sum())

    df = df.dropna()

    print("\nMissing Values After Cleaning:")
    print(df.isnull().sum())

    return df


def remove_duplicates(df):
    """Remove duplicate rows."""

    print("\nDuplicate Rows Before Cleaning:", df.duplicated().sum())

    df = df.drop_duplicates()

    print("Duplicate Rows After Cleaning:", df.duplicated().sum())

    return df


def drop_unnecessary_columns(df):
    """Remove unnecessary columns."""

    if "date" in df.columns:
        df = df.drop("date", axis=1)

    return df


def split_features_target(df):
    """Split features and target."""

    X = df.drop("price", axis=1)
    y = df["price"]

    return X, y


def split_train_test(X, y):
    """Split dataset into training and testing data."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    return X_train, X_test, y_train, y_test