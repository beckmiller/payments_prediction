from lightgbm import LGBMRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from typing import Tuple

from src.data import load_debit
from src.data import split_train_test


def train_test_model(data_path: str) -> Tuple[ForecasterAutoreg, int]:
    data = load_debit(data_path)
    train, test = split_train_test(data)
    test_length = len(test)
    forecaster = ForecasterAutoreg(
        regressor=LGBMRegressor(
            lambda_l1=0.01,
            learning_rate=0.6,
            num_leaves=10,
            max_leaf_nodes=30,
            max_depth=10,
            n_estimators=500,
            random_state=123,
        ),
        lags=10,
    )

    forecaster.fit(y=train["Debit"])

    return forecaster, test_length


def train_dataset_model(data_path: str) -> ForecasterAutoreg:
    data = load_debit(data_path)

    forecaster = ForecasterAutoreg(
        regressor=LGBMRegressor(
            lambda_l2=0.01,
            lambda_l1=0.01,
            learning_rate=0.6,
            num_leaves=32,
            max_leaf_nodes=30,
            max_depth=10,
            n_estimators=2000,
            random_state=123,
        ),
        lags=10,
    )

    forecaster.fit(y=data["Debit"])
    # from joblib import dump, load

    # dump(forecaster, filename='debit_forecaster.py')
    # # forecaster_loaded = load('forecaster.py')
    # # forecaster_loaded.predict(steps=3)
    
    return forecaster
