import numpy as np

from base_naive_bayes import BaseNaiveBayes
from labels import Categories


class NaiveBayes4(BaseNaiveBayes):
    a: int
    category_estimates: dict[Categories, float]
    category_counts: dict[Categories, int]
    word_estimates: dict[Categories, np.ndarray]
    labels: np.ndarray
    features: np.ndarray

    def __init__(self, a: int):
        self.a = a

    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.features = np.where(features != 0, 1, features)

        self.labels = labels

        print("Set percentages of each category...")
        self._set_category_estimates()
        print("Setting sum of words in each category...")
        self._set_category_counts()
        print("Setting word estimates...")
        self._set_word_estimates()

    def _set_category_estimates(self):
        self.category_estimates = {}
        total_count = len(self.labels)
        for category in Categories:
            category_indices = self.labels == category.value
            category_count = np.sum(category_indices)
            category_count = int(category_count)
            self.category_estimates[category] = category_count / total_count

    def _set_word_estimates(self):
        self.word_estimates = {}
        for category in Categories:
            # [0] because np.where returns a tuple
            category_indices = np.where(self.labels == category.value)[0]
            category_features = self.features[category_indices]

            word_occurrences = np.sum(category_features, axis=0)
            numerator = np.add(word_occurrences, self.a)

            denominator = self.category_counts[category] + self.a * 2

            percentages = np.divide(numerator, denominator)
            self.word_estimates[category] = percentages

    def _set_category_counts(self):
        self.category_counts = {}

        for category in Categories:
            category_estimate_ln = np.log(self.category_estimates[category])

            category_indices = self.labels == category.value
            category_count = np.sum(category_indices)
            category_count = int(category_count)

            self.category_counts[category] = category_count

    def predict(self, feature_vector: np.ndarray) -> Categories:
        category_probabilities = {}
        test_features = np.where(feature_vector != 0, 1, feature_vector)
        inverse_test_features = 1 - test_features
        for category in Categories:
            total = 0
            category_probability = self.category_estimates[category]
            category_ln_probability = np.log(category_probability)

            probabilities = self.word_estimates[category] * test_features

            inverse_word_estimates = np.subtract(1, self.word_estimates[category])
            inverse_probability = inverse_word_estimates * inverse_test_features

            probabilities_added = np.add(probabilities, inverse_probability)
            probabilities_added_ln = np.log(probabilities_added)
            total = np.sum(probabilities_added_ln)
            category_probabilities[category] = category_ln_probability + total

        return max(category_probabilities, key=category_probabilities.get)
