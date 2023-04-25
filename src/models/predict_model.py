from typing import List
import numpy as np

from .train_model import train_dataset_model, train_test_model


def train_test_predict(path: str) -> List[float]:
    """
    Trains a model and uses it to predict the next values length of test set.

    Args:
        path: A string containing the dataset location.

    Returns:
        A list of floats containing the predicted values for the next steps in the dataset.
    """
    model, steps = train_test_model(path)
    predicted_values = model.predict(steps)
    predicted_values = np.maximum(predicted_values, 0)
    
    return predicted_values

def train_dataset_predict(path: str) -> List[float]:
    """
    Trains a model on a whole dataset and uses it to predict the next 30 day values.

    Args:
        path: A string containing the dataset location.

    Returns:
        A list of floats containing the predicted values for the next steps in the dataset.
    """
    model = train_dataset_model(path)
    steps = 30
    predicted_values = model.predict(steps)
    predicted_values = np.maximum(predicted_values, 0)

    return predicted_values