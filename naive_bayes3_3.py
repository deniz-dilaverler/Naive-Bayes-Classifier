import numpy as np

from labels import Categories
from naive_bayes3_2 import NaiveBayes2


class NaiveBayes3(NaiveBayes2):
    a: int

    def __init__(self, a: int):
        self.a = a

    def _new_set_category_word_ln_probabilities(self):
        self.category_word_ln_percentages = [None] * len(Categories)
        category_features = {}
        for category in Categories:
            category_indices = np.where(self.labels == category.value)[0]
            features = self.features[category_indices]
            category_features[category] = features

        for category in Categories:
            features = category_features[category]
            column_sum = np.sum(features, axis=0)

            numerator = np.add(column_sum, self.a)
            feature_length = len(features[0])

            denominator = self.category_total_word_counts[category] + self.a * feature_length

            percentages = np.divide(numerator, denominator)
            ln_percentages = np.log(percentages)
            self.category_word_ln_percentages[category.value] = ln_percentages

    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.features = features
        self.labels = labels

        print("Setting total words in each category...")
        self._set_category_total_word_counts()
        print("Set percentages of each category...")
        self._set_category_estimates()
        print("Setting sum of words in each category...")
        self._new_set_category_word_ln_probabilities()
