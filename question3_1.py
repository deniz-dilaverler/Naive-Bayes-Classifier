import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
from matplotlib import pyplot as plt

from labels import Categories

y_train_csv_path = 'dataset/y_train.csv'
y_test_csv_path = 'dataset/y_test.csv'

x_test_csv_path = 'dataset/x_test.csv'
x_train_csv_path = 'dataset/x_train.csv'

def create_category_pie_chart(labels: list[str], percentages: list[float]):
    fig1, ax1 = plt.subplots()

    ax1.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')

    plt.show()


def get_category_counts(input_csv_path: str):
    input_data = pd.read_csv(input_csv_path)
    category_counts = input_data.value_counts()
    category_counts = category_counts.sort_index()

    return category_counts
def get_result_category_percentages(input_csv_path: str):
    category_counts = get_category_counts(input_csv_path)

    total_count = sum(category_counts)
    percentages = [count / total_count for count in category_counts]
    labels = []
    for category_id in category_counts.index:
        category_id = category_id[0] # for some reason, category_id is a tuple
        category_name = Categories(category_id).name
        labels.append(category_name)

    return (labels, percentages)


def get_category_distribution_pie_chart(input_csv_path:str):
    labels, percentages = get_result_category_percentages(input_csv_path)
    create_category_pie_chart(labels, percentages)


#comment out one of the following lines to see the pie chart for
# the training or testing set
#get_category_distribution_pie_chart(y_train_csv_path)
#get_category_distribution_pie_chart(y_test_csv_path)


def get_category_words(
        labels: DataFrame,
        features: DataFrame,
        category: Categories
) -> DataFrame:
    category_indices = labels[labels[0] == category.value].index

    category_words = features.iloc[category_indices]
    return category_words


def get_total_word_count(
        category_words: DataFrame,
) -> int:
    category_words_matrix = category_words.to_numpy()
    row_sums = np.sum(category_words_matrix, axis=1)
    total_word_count = np.sum(row_sums, axis=0)
    total_word_count = int(total_word_count)
    return total_word_count


def get_usage_of_words(
        words_df: DataFrame,
        word: str
) -> int:
    word_index = words_df.columns.get_loc(word)
    word_count = np.sum(words_df.iloc[:, word_index])
    return int(word_count)


def question_3_1_d(
    labels_path: str,
    features_path: str,
    category: Categories,
    word: str
):
    labels = pd.read_csv(labels_path, header=None)
    features = pd.read_csv(features_path, delimiter=" ")

    category_words = get_category_words(
        labels=labels,
        features=features,
        category=category
    )
    total_word_count = get_total_word_count(category_words)
    word_count = get_usage_of_words(category_words, word)
    probability = word_count / total_word_count
    logarithmic_probability = np.log(probability)
    print(f"Total word count: {total_word_count}")
    print(f"Count of the word '{word}': {word_count}")
    print(f"Logarithmic probability of '{word}': {logarithmic_probability}")


words = ("alien", "thunder")
category = Categories.TECH

for word in words:
    question_3_1_d(
        labels_path=y_train_csv_path,
        features_path=x_train_csv_path,
        category=category,
        word=word
    )
