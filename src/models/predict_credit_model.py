from typing import List
import numpy as np

from .train_credit_model import credit_train_test_model, credit_dataset_model


def credit_train_test_predict(path: str) -> List[float]:
    """
    Trains a model and uses it to predict the next values length of test set.

    Args:
        path: A string containing the dataset location.

    Returns:
        A list of floats containing the predicted values for the next steps in the dataset.
    """
    model, steps = credit_train_test_model(path)
    predicted_values = model.predict(steps=steps)
    predicted_values = np.maximum(predicted_values, 0)
    
    return predicted_values

def credit_dataset_predict(path: str) -> List[float]:
    """
    Trains a model on a whole dataset and uses it to predict the next 30 day values.

    Args:
        path: A string containing the dataset location.

    Returns:
        A list of floats containing the predicted values for the next steps in the dataset.
    """
    model = credit_dataset_model(path)
    steps = 30
    predicted_values = model.predict(steps)
    predicted_values = np.maximum(predicted_values, 0)

    return predicted_values