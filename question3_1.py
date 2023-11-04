import pandas as pd
from matplotlib import pyplot as plt

from labels import Categories


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



y_train_csv_path = 'dataset/y_train.csv'
y_test_csv_path = 'dataset/y_test.csv'

#comment out one of the following lines to see the pie chart for 
# the training or testing set
#get_category_distribution_pie_chart(y_train_csv_path)
get_category_distribution_pie_chart(y_test_csv_path)
