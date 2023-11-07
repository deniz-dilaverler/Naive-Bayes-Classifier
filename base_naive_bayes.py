import datetime
from abc import ABC, abstractmethod

import numpy as np

from confusion_matrix import CategoryConfusionMatrix
from labels import Categories


class BaseNaiveBayes(ABC):
    @abstractmethod
    def fit(self, features: np.ndarray, labels: np.ndarray):
        pass

    @abstractmethod
    def predict(self, test_features: np.ndarray):
        pass

    def test(self, test_features: np.ndarray, test_labels: np.ndarray) -> float:
        correct = 0
        confusion_matrix = CategoryConfusionMatrix()
        start_ts = datetime.datetime.now().timestamp()
        for i, test_vector in enumerate(test_features):
            ts = datetime.datetime.now().timestamp()
            print(f"Testing {i}...")
            prediction = self.predict(test_vector)

            print(f"Prediction: {prediction}")
            actual_category = Categories(test_labels[i])

            confusion_matrix.add_prediction(prediction, actual_category)
            print(f"Actual: {actual_category}")
            print("----------")
            if prediction == actual_category:
                correct += 1
        print(f"Correct: {correct}")
        print(f"Total: {len(test_features)}")
        print("---------------")

        finish_ts = datetime.datetime.now().timestamp()
        time_taken = finish_ts - start_ts

        print(f"Time taken: {time_taken}s")
        print("---------------")

        print("Confusion Matrix")
        print(confusion_matrix)
        return correct / len(test_features)
