import numpy as np
import pandas as pd

from naive_bayes3_2 import NaiveBayes2
from naive_bayes3_3 import NaiveBayes3


def get_features(csv_path: str) -> np.ndarray:
    input_data = pd.read_csv(csv_path, delimiter=" ").to_numpy()
    return input_data


def get_labels(csv_path: str) -> np.ndarray:
    input_data = pd.read_csv(csv_path, header=None).to_numpy()
    return input_data


y_train_csv_path = 'dataset/y_train.csv'
y_test_csv_path = 'dataset/y_test.csv'

x_test_csv_path = 'dataset/x_test.csv'
x_train_csv_path = 'dataset/x_train.csv'

y_train = get_labels(y_train_csv_path)
x_train = get_features(x_train_csv_path)

x_test = get_features(x_test_csv_path)
y_test = get_labels(y_test_csv_path)

print("Datasets are loaded")
"""
naive_bayes3_2 = NaiveBayes2()
naive_bayes3_2.fit(x_train, y_train)
error_rate = naive_bayes3_2.test(x_test, y_test)

print(f"Question3.2 error rate: {error_rate}")
"""

naive_bayes3_3 = NaiveBayes3(a=1)
naive_bayes3_3.fit(x_train, y_train)
error_rate = naive_bayes3_3.test(x_test, y_test)

print(f"Question3.3 error rate: {error_rate}")


