from fastapi import FastAPI
import uvicorn

from src.models import train_test_predict, train_dataset_predict
from src.models import credit_train_test_predict, credit_dataset_predict
from src.data import save_cleaned_data, save_debit_train, cleaned_data, train_read, test_read
from src.data import load_credit, save_credit_train, cleaned_credit_data
from src.visualization import plot_future, plot_train_test

from api import forecast

app = FastAPI()
app.include_router(forecast.router)

DEBIT_INPUT_PATH = "data\\external\\UAE_Categories.xlsx"
CREDIT_INPUT_PATH = "data\\external\\UAE_Categories.xlsx"

DEBIT_TRAIN_PATH = "data\\raw\\train_debit.csv"
DEBIT_TEST_PATH = "data\\raw\\test_debit.csv"

DEBIT_OUTPUT_PATH = "data\processed"

TRAIN_TEST_OUTPUT_PATH = "data\\raw"

debit_original_data = cleaned_data(DEBIT_INPUT_PATH)
debit_forecast = train_dataset_predict(DEBIT_INPUT_PATH)
debit_prediction = train_test_predict(DEBIT_INPUT_PATH)
credit_forecast = credit_dataset_predict(CREDIT_INPUT_PATH)
credit_original_data = cleaned_credit_data(CREDIT_INPUT_PATH)

# train_data = train_read(DEBIT_TRAIN_PATH)
# train_data = test_read(DEBIT_TEST_PATH)

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
    
    # plot the forecast for next 30 days
    # print(debit_original_data.to_dict())
    # plot_future(debit_original_data, debit_forecast, title='Debit')
    # plot_future(credit_original_data, credit_forecast, title='Credit')
    
    # save_cleaned_data(DEBIT_INPUT_PATH, DEBIT_OUTPUT_PATH)
    # save_debit_train(DEBIT_INPUT_PATH, TRAIN_TEST_OUTPUT_PATH)
    # save_credit_train(CREDIT_INPUT_PATH, TRAIN_TEST_OUTPUT_PATH)
    # plot_train_test(train_data, train_data, debit_prediction)
    # print(credit_train_test_predict(CREDIT_INPUT_PATH))
    # print(train_test_predict(DEBIT_INPUT_PATH))
    # print(train_dataset_predict(DEBIT_INPUT_PATH))
