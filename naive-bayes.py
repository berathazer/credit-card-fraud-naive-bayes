import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fillMissingValuesWithMean(data, featureLength):
    """Fill all the missing values with mean values """
    for i in range(featureLength):
        if is_binary_column_without_nan(data, i):
            data = fill_binary_column(data, i)
        else:
            data = calculate_mean(data, i)
    return data


def is_binary_column_without_nan(data, index):
    """Check the column is  binary or not"""
    column_data = data.iloc[:, index]

    column_type = column_data.dtype

    if column_type in ['int64', 'float64']:
        unique_values = column_data.dropna().unique()
        binary = all(value in [0, 1] for value in unique_values)
        return binary

    return False


def calculate_mean(data, index):
    """Calculates the mean of the given data"""
    column_data = data.iloc[:, index]
    mean_value = column_data.mean()
    data.iloc[:, index].fillna(mean_value, inplace=True)
    return data


def fill_binary_column(data, index):
    """fill binary column of given data with majority values"""
    column_data = data.iloc[:, index]
    majority_value = column_data.mode().iloc[0]
    data.iloc[:, index].fillna(majority_value, inplace=True)

    return data


def calculate_min_max_normalization(data, index):
    """Calculate Min Max Normalization"""
    column_data = data.iloc[:, index]

    # Min-Max Normalization: (X - min) / (max - min)
    min_value = column_data.min()
    max_value = column_data.max()

    normalized_data = (column_data - min_value) / (max_value - min_value)

    data.iloc[:, index] = normalized_data

    return data


def normalization(data):
    """Normalize the data by calling the normalization function"""
    for i in range(featureLength):
        if is_binary_column_without_nan(data,i):
            continue
        data = calculate_min_max_normalization(data,i)
    return data


def kfold_cross_validation(data, k=5):
    """Kfold cross validation"""

    X = data.iloc[:, :-2]
    y = data.iloc[:, -2]

    data_length = len(data)

    fold_size = data_length // k

    accuracies = []

    for i in range(k):

        fold_start = i * fold_size
        fold_end = min((i + 1) * fold_size, data_length)

        test_indices = range(fold_start, fold_end)
        train_indices = [j for j in range(data_length) if j not in test_indices]

        X_train = X.iloc[train_indices, :]
        y_train = y.iloc[train_indices]
        X_test = X.iloc[test_indices, :]
        y_test = y.iloc[test_indices]

        print(len(X_train),len(y_train))


path = './card_transdata.xlsx'

dataset = pd.read_excel(path, sheet_name="card_transdata")

featureLength = len(dataset.columns) - 1

dataLength = len(dataset)

dataWithoutNormalization = fillMissingValuesWithMean(dataset, featureLength)

dataWithNormalization = normalization(dataWithoutNormalization)

kfold_cross_validation(dataWithoutNormalization, 5)


