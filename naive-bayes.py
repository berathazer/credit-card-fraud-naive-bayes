import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

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
    normalized_data = data.copy()

    for i in range(featureLength):
        if is_binary_column_without_nan(normalized_data, i):
            continue
        normalized_data = calculate_min_max_normalization(normalized_data, i)
    return normalized_data


def train_naive_bayes(X_train, y_train):
    """Naive Bayes modelini eğitim verisi ile oluştur"""
    class_probabilities = defaultdict(float)
    total_samples = len(y_train)

    for class_label in set(y_train):
        class_probabilities[class_label] = sum(y_train == class_label) / total_samples

    feature_probabilities = defaultdict(dict)

    for feature in X_train.columns:
        for class_label in set(y_train):
            feature_probabilities[feature][class_label] = (
                sum((X_train[feature] == 1) & (y_train == class_label)) + 1
            ) / (sum(y_train == class_label) + 2)

    return {"class_probabilities": class_probabilities, "feature_probabilities": feature_probabilities}

def predict_naive_bayes(X_test, model):
    """Naive Bayes modeli ile test verisi üzerinde tahmin yap"""
    predictions = []

    for _, row in X_test.iterrows():
        class_scores = defaultdict(float)

        for class_label, class_probability in model["class_probabilities"].items():
            feature_score = 0

            for feature, value in row.items():
                feature_score += model["feature_probabilities"][feature][class_label] if value == 1 else 1 - model["feature_probabilities"][feature][class_label]

            class_scores[class_label] = feature_score + class_probability

        predicted_class = max(class_scores, key=class_scores.get)
        predictions.append(predicted_class)

    return predictions

def calculate_accuracy(y_true, y_pred):
    """Doğruluk (accuracy) ölçümünü hesapla"""
    correct_predictions = sum(y_true == y_pred)
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples
    return accuracy

def kfold_cross_validation(data, k=5):
    """Kfold cross validation"""

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    data_length = len(data)

    fold_size = data_length // k
    accuracies = []

    for i in range(k):

        fold_start = i * fold_size
        fold_end = min((i + 1) * fold_size, data_length)

        test_indices = range(fold_start, fold_end)
        train_indices = [j for j in range(data_length) if j not in test_indices]

        X_train, X_test = X.iloc[train_indices, :], X.iloc[test_indices, :]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        print(X_train)
        trainModel = train_naive_bayes(X_train, y_train)
        prediction = predict_naive_bayes(X_test, trainModel)
        accuracy = calculate_accuracy(y_test, prediction)
        accuracies.append(accuracy)
    return accuracies








path = './card_transdata.xlsx'
#path = './veriler.xlsx'
#path = './card_transdata.xlsx'

dataset = pd.read_excel(path, sheet_name="card_transdata")

featureLength = len(dataset.columns) - 1

dataLength = len(dataset)

dataWithoutNormalization = fillMissingValuesWithMean(dataset, featureLength)

dataWithNormalization = normalization(dataWithoutNormalization)

accuracyWithoutNormalization = kfold_cross_validation(dataWithoutNormalization, 5)

accuracyWithNormalization = kfold_cross_validation(dataWithNormalization, 5)

print("Accuracy without Normalization:",accuracyWithoutNormalization)

print("Accuracy with Normalization:",accuracyWithNormalization)


# Grafik için doğruluk oranlarını ayarla
methods = ['Normalization Before'] * len(accuracyWithoutNormalization) + ['Normalization After'] * len(accuracyWithNormalization)
accuracies = accuracyWithoutNormalization + accuracyWithNormalization

plt.bar(methods, accuracies, color=['blue', 'green'])

# Grafik başlığı ve etiketleri ekle
plt.title('Naive Bayes Çapraz Doğrulama Sonuçları')
plt.xlabel('Normalleştirme Yöntemi')
plt.ylabel('Doğruluk Oranı')

# Grafiği göster
plt.show()