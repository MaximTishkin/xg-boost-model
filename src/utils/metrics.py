from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def calculate_metrics(model, x_test, y_test):
    predictions = model.predict(x_test)
    return {
        'R2': r2_score(y_test, predictions),
        'MAE': mean_absolute_error(y_test, predictions),
        'MSE': mean_squared_error(y_test, predictions)
    }
