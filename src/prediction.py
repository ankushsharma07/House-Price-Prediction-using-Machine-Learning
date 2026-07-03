def predict_house_price(model, X_test):
    """
    Predict house prices using the trained Linear Regression model.
    """

    predictions = model.predict(X_test)

    return predictions