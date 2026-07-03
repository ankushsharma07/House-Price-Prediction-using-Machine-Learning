from src.data_cleaning import *
from src.train_model import *
from src.prediction import *

# =====================================
# Load and Clean Data
# =====================================

df = load_data()

df = handle_missing_values(df)

df = remove_duplicates(df)

df = drop_unnecessary_columns(df)

# =====================================
# Split Features and Target
# =====================================

X, y = split_features_target(df)

# =====================================
# Train-Test Split
# =====================================

X_train, X_test, y_train, y_test = split_train_test(X, y)

# =====================================
# Train Linear Regression Model
# =====================================

print("=" * 50)
print("LINEAR REGRESSION MODEL")
print("=" * 50)

linear_model = train_linear_regression(X_train, y_train)

# =====================================
# Evaluate Model
# =====================================

evaluate_model(linear_model, X_test, y_test)

# =====================================
# Predict House Prices
# =====================================

print("\n" + "=" * 50)
print("SAMPLE PREDICTIONS")
print("=" * 50)

predictions = predict_house_price(linear_model, X_test)

print(predictions[:10])