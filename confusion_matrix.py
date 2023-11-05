import numpy as np

from labels import Categories


# rows are predictions
# columns are actual
class CategoryConfusionMatrix:
    matrix: np.ndarray

    def __init__(self):
        self.matrix = np.zeros((len(Categories), len(Categories)), dtype=int)

    def add_prediction(self, predicted: Categories, actual: Categories):
        self.matrix[predicted.value][actual.value] += 1

    def __str__(self):
        return str(self.matrix)
