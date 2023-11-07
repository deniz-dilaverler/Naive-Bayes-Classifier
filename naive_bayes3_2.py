import numpy as np

from base_naive_bayes import BaseNaiveBayes
from labels import Categories


# Naive Bayes Classifier for Question3.2
class NaiveBayes2(BaseNaiveBayes):
    features: np.ndarray
    labels: np.ndarray
    category_total_word_counts: dict[Categories, int]
    category_estimates: dict[Categories, float]
    category_word_ln_percentages: np.ndarray

    def _get_category_word_count(self, category: Categories) -> int:
        return self.category_total_word_counts[category]

    def _set_category_total_word_counts(self):
        self.category_total_word_counts = {}

        for category in Categories:
            category_indices = np.where(self.labels == category.value)[0]
            category_features = self.features[category_indices]

            row_sums = np.sum(category_features, axis=1)
            total_word_count = np.sum(row_sums, axis=0)
            total_word_count = int(total_word_count)
            self.category_total_word_counts[category] = total_word_count

    def _set_category_estimates(self):
        self.category_estimates = {}
        total_count = np.sum(self.labels)
        for category in Categories:
            category_indices = self.labels == category.value
            category_count = np.sum(category_indices)
            category_count = int(category_count)
            self.category_estimates[category] = category_count / total_count

    def _set_category_word_ln_percentages(self):
        self.category_word_ln_percentages = [None] * len(Categories)
        category_features = {}
        for category in Categories:
            category_indices = np.where(self.labels == category.value)[0]
            features = self.features[category_indices]
            category_features[category] = features

        for category in Categories:
            features = category_features[category]
            column_sum = np.sum(features, axis=0)
            percentages = np.divide(column_sum, self.category_total_word_counts[category])
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
        self._set_category_word_ln_percentages()

    def predict(self, test_features: np.ndarray):

        category_probabilities = {}
        for category in Categories:
            total = 0
            category_probability = self.category_estimates[category]
            category_ln_probability = np.log(category_probability)

            vector_multiplied = test_features * self.category_word_ln_percentages[category.value]

            total = np.sum(vector_multiplied)
            category_probabilities[category] = category_ln_probability + total

        return max(category_probabilities, key=category_probabilities.get)
