from src.data_cleaning import *
from src.train_model import *
from src.prediction import *

# Load and Clean Data

df = load_data()

df = handle_missing_values(df)

df = remove_duplicates(df)

df = drop_unnecessary_columns(df)

# Split Features and Target

X, y = split_features_target(df)

# Train-Test Split

X_train, X_test, y_train, y_test = split_train_test(X, y)

# Train Linear Regression Model

print("=" * 50)
print("LINEAR REGRESSION MODEL")
print("=" * 50)

linear_model = train_linear_regression(X_train, y_train)

# Evaluate Model

evaluate_model(linear_model, X_test, y_test)

# Take User Input

print("\n" + "=" * 50)
print("HOUSE PRICE PREDICTION")
print("=" * 50)

bedrooms = float(input("Enter number of bedrooms: "))
bathrooms = float(input("Enter number of bathrooms: "))
sqft_living = float(input("Enter sqft living: "))
sqft_lot = float(input("Enter sqft lot: "))
floors = float(input("Enter number of floors: "))
waterfront = float(input("Enter waterfront (0 = No, 1 = Yes): "))
view = float(input("Enter view rating: "))
condition = float(input("Enter condition rating: "))

# Create input data
user_input = [[
    bedrooms,
    bathrooms,
    sqft_living,
    sqft_lot,
    floors,
    waterfront,
    view,
    condition
]]

# Prediction
predicted_price = linear_model.predict(user_input)

print("\n" + "=" * 50)
print("PREDICTED HOUSE PRICE")
print("=" * 50)

print(f"Predicted Price: ${predicted_price[0]:,.2f}")
