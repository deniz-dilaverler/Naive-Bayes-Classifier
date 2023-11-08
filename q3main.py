import numpy as np
import pandas as pd
import sys
import os
from dotenv import load_dotenv

from naive_bayes3_2 import NaiveBayes2
from naive_bayes3_3 import NaiveBayes3
from naive_bayes3_4 import NaiveBayes4

load_dotenv()

def get_features(csv_path: str) -> np.ndarray:
    input_data = pd.read_csv(csv_path, delimiter=" ").to_numpy()
    return input_data


def get_labels(csv_path: str) -> np.ndarray:
    input_data = pd.read_csv(csv_path, header=None).to_numpy()
    return input_data


def get_env_var(var_name: str) -> str:
    var = os.environ.get(var_name)
    if var is None:
        raise ValueError(f"Environment variable {var_name} is not set")

    return var


y_train_csv_path = get_env_var('Y_TRAIN_CSV_PATH')
y_test_csv_path = get_env_var('Y_TEST_CSV_PATH')

x_test_csv_path = get_env_var('X_TEST_CSV_PATH')
x_train_csv_path = get_env_var('X_TRAIN_CSV_PATH')

print("Loading datasets...")
y_train = get_labels(y_train_csv_path)
x_train = get_features(x_train_csv_path)

x_test = get_features(x_test_csv_path)
y_test = get_labels(y_test_csv_path)

print("Datasets are loaded")


def question_3_2():
    naive_bayes3_2 = NaiveBayes2()
    naive_bayes3_2.fit(x_train, y_train)
    error_rate = naive_bayes3_2.test(x_test, y_test)

    print(f"Question3.2 accuracy: {error_rate}")


def question3_3():
    naive_bayes3_3 = NaiveBayes3(a=1)
    naive_bayes3_3.fit(x_train, y_train)
    error_rate = naive_bayes3_3.test(x_test, y_test)

    print(f"Question3.3 accuracy: {error_rate}")


def question3_4():
    naive_bayes3_4 = NaiveBayes4(a=1)
    naive_bayes3_4.fit(x_train, y_train)
    error_rate = naive_bayes3_4.test(x_test, y_test)

    print(f"Question3.4 accuracy: {error_rate}")


def main(argument_count):
    if argument_count == 2:
        question_3_2()
    elif argument_count == 3:
        question3_3()
    elif argument_count == 4:
        question3_4()
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
