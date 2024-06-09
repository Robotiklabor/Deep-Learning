import numpy as np

def load_csv():
    # Load the data
    x_data = np.loadtxt('X_train.csv', delimiter=',', dtype=np.float32)
    y_data = np.loadtxt('Y_train.csv', delimiter=',', dtype=np.float32)

    x_data = np.transpose(x_data)  # Transpose to shape (5000, 3)
    y_data = np.transpose(y_data)  # Transpose to shape (5000, 3)

    # Standardize the data
    x_data_scaled = (x_data - np.mean(x_data, axis=0)) / np.std(x_data, axis=0)
    y_data_scaled = (y_data - np.mean(y_data, axis=0)) / np.std(y_data, axis=0)

    return x_data_scaled, y_data_scaled
