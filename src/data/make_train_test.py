

def split_train_test(data):
    """
    Splits the data into train and test sets.
    Returns trian, test sets
    """
    train = data.loc[data.index < '2022-04-20']
    test = data.loc[data.index >= '2022-04-20']

    return train, test