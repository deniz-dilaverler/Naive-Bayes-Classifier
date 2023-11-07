import numpy as np
import pandas as pd
import sys

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


def question_3_2():
    naive_bayes3_2 = NaiveBayes2()
    naive_bayes3_2.fit(x_train, y_train)
    error_rate = naive_bayes3_2.test(x_test, y_test)

    print(f"Question3.2 error rate: {error_rate}")

def question3_3():
    naive_bayes3_3 = NaiveBayes3(a=1)
    naive_bayes3_3.fit(x_train, y_train)
    error_rate = naive_bayes3_3.test(x_test, y_test)

    print(f"Question3.3 error rate: {error_rate}")


def main(argument_count):
    if argument_count == 2:
        question_3_2()
    elif argument_count == 3:
        question3_3()
    elif argument_count == 4:
        raise NotImplementedError()
    else:
        print("Invalid argument count")


if len(sys.argv) != 2:
    print("Usage: python main_module.py [2|3|4]")
    sys.exit(1)

try:
    argument_val = int(sys.argv[1])
except ValueError:
    print("Invalid argument. Please provide 2, 3, or 4.")
    sys.exit(1)

main(argument_val)
