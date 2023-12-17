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
    """Min-Max normalizasyonunu hesaplayan fonksiyon"""
    column_data = data.iloc[:, index]
    min_value = column_data.min()
    max_value = column_data.max()

    normalized_data = (column_data - min_value) / (max_value - min_value)

    data.iloc[:, index] = normalized_data

    return data


def normalization(data):
    """Normalize the data by calling the normalization function"""

    for i in range(featureLength):
        if is_binary_column_without_nan(data, i):
            continue
        data = calculate_min_max_normalization(data, i)
    return data



def train_naive_bayes(X_train, y_train):
    """Naive Bayes modelini eğitim verisi ile oluştur.

    Args:
        X_train (pd.DataFrame): Eğitim verisi özellikleri.
        y_train (pd.Series): Eğitim verisi sınıf etiketleri.

    Returns:
        dict: Eğitilmiş Naive Bayes modeli.
    """
    class_probabilities = defaultdict(float)
    total_samples = len(y_train)

    # Sınıf olasılıklarının hesaplanması
    for class_label in set(y_train):
        class_probabilities[class_label] = sum(y_train == class_label) / total_samples

    feature_probabilities = defaultdict(dict)

    # Özellik olasılıklarının hesaplanması
    for feature in X_train.columns:
        for class_label in set(y_train):
            # Özellik olasılıklarının hesaplanması
            feature_probabilities[feature][class_label] = (
                (X_train[feature] == 1).sum() & (y_train == class_label).sum() + 1
            ) / (sum(y_train == class_label) + 2)

    return {"class_probabilities": class_probabilities, "feature_probabilities": feature_probabilities}


def predict_naive_bayes(X_test, model):
    """Naive Bayes modeli ile test verisi üzerinde tahmin yap.

    Args:
        X_test (pd.DataFrame): Test verisi özellikleri.
        model (dict): Eğitilmiş Naive Bayes modeli.

    Returns:
        list: Tahmin edilen sınıf etiketleri.
    """
    predictions = []

    for _, row in X_test.iterrows():
        class_scores = defaultdict(float)

        for class_label, class_probability in model["class_probabilities"].items():
            feature_score = 0

            for feature, value in row.items():
                # Özellik olasılıklarına göre puanlama
                feature_score += model["feature_probabilities"][feature][class_label] if value == 1 else 1 - model["feature_probabilities"][feature][class_label]

            # Sınıfın toplam puanı
            class_scores[class_label] = feature_score + class_probability

        # En yüksek puanlı sınıfı seçme
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
        trainModel = train_naive_bayes(X_train, y_train)
        prediction = predict_naive_bayes(X_test, trainModel)
        accuracy = calculate_accuracy(y_test, prediction)
        accuracies.append(accuracy)
        print(f"{i+1}. KFold: %{accuracy}")
    return accuracies








path = './card_transdataV2.xlsx'
#path = './veriler.xlsx'
#path = './diabetes1.xlsx'
#path = "./IRIS.xlsx"

dataset = pd.read_excel(path)

featureLength = len(dataset.columns) - 1

dataLength = len(dataset)

k = 5

dataWithoutNormalization = fillMissingValuesWithMean(dataset.copy(), featureLength)

dataWithNormalization = normalization(dataWithoutNormalization.copy())

print("Unnormalized Data\n")
accuracyWithoutNormalization = kfold_cross_validation(dataWithoutNormalization.copy(), k)

print("\nNormalized Data\n")
accuracyWithNormalization = kfold_cross_validation(dataWithNormalization.copy(), k)


print("\nAccuracy without Normalization:",accuracyWithoutNormalization)

print("Accuracy with Normalization:",accuracyWithNormalization)

print("\nMean Accuracy With Normalization:",np.mean(accuracyWithNormalization))
print("Mean Accuracy Without Normalization:",np.mean(accuracyWithoutNormalization))


methods = ['Normalization Before'] * len(accuracyWithoutNormalization) + ['Normalization After'] * len(accuracyWithNormalization)
accuracies = accuracyWithoutNormalization + accuracyWithNormalization

fig, axs = plt.subplots(2, 1, figsize=(8, 8))

# İlk subplot: Normalizasyondan Önce
axs[0].plot(range(1, k + 1), accuracyWithoutNormalization, marker='o', linestyle='-', color='blue')
axs[0].set_title('Normalizasyondan Önce')
axs[0].set_ylabel('Doğruluk Oranı')
axs[0].set_xticks(range(1, k + 1))

# İkinci subplot: Normalizasyondan Sonra
axs[1].plot(range(1, k + 1), accuracyWithNormalization, marker='o', linestyle='-', color='green')
axs[1].set_title('Normalizasyondan Sonra')

axs[1].set_ylabel('Doğruluk Oranı')
axs[1].set_xticks(range(1, k + 1))


# Alt grafikler arasında boşluk bırak
plt.tight_layout()

# Grafiği göster
plt.show()