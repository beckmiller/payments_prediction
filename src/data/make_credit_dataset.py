import pandas as pd
from .make_train_test import split_train_test


def load_credit(data_path):
    """ Load data from a given path"""
    df = pd.read_excel(data_path,  sheet_name="Credits")
    df = df[["Date", "Credit"]]
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
    df.Credit = df.Credit.astype(float)
    credit_by_date = df.groupby('Date')['Credit'].sum().reset_index()
    credit_by_date = credit_by_date[credit_by_date['Credit'] < 900000]
    # Group the DataFrame by Date and sum the Debit column for each date
    data_range = pd.date_range(start='19/12/2021', end='20/05/2022', freq='D')
    final_df = credit_by_date.set_index('Date').reindex(data_range, fill_value=0)
    
    return final_df

def save_credit_train(input_path, output_path):
    debit_data = load_credit(input_path)
    train_data, test_data = split_train_test(debit_data)
    train_data.to_csv(f'{output_path}\\train_credit.csv')
    test_data.to_csv(f'{output_path}\\test_credit.csv')

def cleaned_credit_data(input_path):
    prepared_data = load_credit(input_path)

    return prepared_data