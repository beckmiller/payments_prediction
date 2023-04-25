import matplotlib.pyplot as plt


def plot_future(original_data_path, future_data, title=None):
    plt.plot(original_data_path, label='Original data')
    plt.plot(future_data, label='Forecast for next 30 days')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_train_test(train_path, test_path, prediction_path):
    plt.plot(train_path, label='Train')
    plt.plot(test_path, label='test')
    plt.plot(prediction_path, label='prediction')
    plt.legend()
    plt.show()
