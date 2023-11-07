import numpy as np

from labels import Categories


class NaiveBayes4:
    a: int
    category_estimates: dict[Categories, float]
    word_estimates: dict[Categories, np.ndarray]
    labels: np.ndarray
    features: np.ndarray

    def __init__(self, a: int):
        self.a = a

    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.features = features
        self.labels = labels

        print("Set percentages of each category...")
        self._set_category_estimates()
    def _set_category_estimates(self):
        self.category_estimates = {}
        total_count = np.sum(self.labels)
        for category in Categories:
            category_indices = self.labels == category.value
            category_count = np.sum(category_indices)
            category_count = int(category_count)
            self.category_estimates[category] = category_count / total_count

    def _set_word_estimates(self):
        self.word_estimates = {}
        for category in Categories:
            category_indices = self.labels == category.value
            category_features = self.features[category_indices]

            numerator = np.add(row_sums, self.a)
            feature_length = len(category_features[0])

            denominator = total_word_count + self.a * feature_length

            percentages = np.divide(numerator, denominator)
            ln_percentages = np.log(percentages)
            self.word_estimates[category] = ln_percentages
