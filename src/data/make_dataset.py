import pandas as pd
from .make_train_test import split_train_test


def load_debit(data_path):
    """ Load data from a given path. Interpolate missing data"""
    df = pd.read_excel(data_path, sheet_name='Debits')
    df = df[["Date", "Debit"]]
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
    # Group the DataFrame by Date and sum the Debit column for each date
    debit_by_date = df.groupby('Date')['Debit'].sum().reset_index()
    date_range = pd.date_range(start='19/12/2021', end='20/05/2022')
    debit_by_date = debit_by_date.set_index('Date').reindex(date_range, fill_value=None)
    final_df = debit_by_date.interpolate()

    return final_df

def save_cleaned_data(input_path, output_path):
    prepared_data = load_debit(input_path)
    prepared_data.to_csv(f'{output_path}\prepared_debit.csv')
    
def save_debit_train(input_path, output_path):
    debit_data = load_debit(input_path)
    train_data, test_data = split_train_test(debit_data)
    train_data.to_csv(f'{output_path}\\train_debit.csv')
    test_data.to_csv(f'{output_path}\\test_debit.csv')

def cleaned_data(input_path):
    prepared_data = load_debit(input_path)

    return prepared_data

def train_read(input_path):
    train = pd.read_csv(input_path, index_col=0)

    return train

def test_read(input_path):
    test = pd.read_csv(input_path, index_col=0)
    
    return test