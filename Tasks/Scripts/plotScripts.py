import matplotlib.pyplot as plt
import math
import numpy as np

label_def = {
    -1: 'All Classes',
    0: 'speed limit 20',
    1: 'speed limit 30',
    2: 'speed limit 50',
    3: 'speed limit 60',
    4: 'speed limit 70',
    5: 'left turn',
    6: 'right turn',
    7: 'beware pedestrian crossing',
    8: 'beware children',
    9: 'beware cycle route ahead'
}


g_labels = [label_def.get(i) for i in range(-1, 10)]

def plot_images(data, n=5):
    """Generate n images from the dataset 

    :param data: matrix of square images
    :type data: numpy.array
    :param n: number of images to print, defaults to 5
    :type n: int, optional
    """

    d = int(round(math.sqrt(data.shape[1]), 0))
    if n > data.shape[0]:
        n = data.shape[0]

    for i in range(n):
        row = data[i]
        image = row.reshape(d, d)
        plt.subplot(1, n, i+1)
        plt.axis('off')
        plt.imshow(image, cmap='gray')


def unzip_scores(score):
    """unzip scores into 2 seperate lists

    :param score: list of score tuples
    :type score: list(float, float)
    :return: 2 numpy arrays
    :rtype: (np.array, np.array)
    """
    return np.array([i for i, j in score]), np.array([j for i, j in score])

def convert_percentage(score_tuple):
    """multiply the given numpy arrays in the tuple by 100

    :param score_tuple: np.array of scores within tuple
    :type score_tuple: (np.array, np.array)
    :return: scores * 100
    :rtype: (np.array, np.array)
    """
    return score_tuple[0] * 100, score_tuple[1] * 100


def plot_scores(scores, group_labels, title='Scores for each classifier', bar_width=0.15, labels=g_labels, figure_size=(14, 8)):
    """Plot the contents of scores as a bar graph

    :param scores: A list containing lists of values to plot
    :type scores: list(list(number))
    :param group_labels: A list of labels for each sub-element in scores
    :type group_labels: list(string)
    :param title: Set the title of the figure, defaults to 'Scores for each classifier'
    :type title: str, optional
    :param bar_width: define the width of the bars, defaults to 0.15
    :type bar_width: float, optional
    :param labels: Specify the x-axis labels, defaults to g_labels
    :type labels: list(string), optional
    :param figure_size: size of the figure, defaults to (14, 8)
    :type figure_size: tuple, optional
    """
    num_groups = len(group_labels)
    x = np.arange(len(labels))
    
    fig, ax = plt.subplots(figsize=figure_size)
    score_plots = []
    for i in range(num_groups):
        bar = ax.bar(x + (bar_width * i), scores[i], bar_width, label=group_labels[i])
        score_plots.append(bar)

    ax.set_title(title)
    ax.set_ylabel('% Accuracy')
    ax.set_xticks(x + (bar_width * math.floor((num_groups / 2))))
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()

    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    def autolabel(bar_group):
        """Attach a text label above each bar in *bar_group*, displaying its height."""
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate('{}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    for i in range(num_groups):
        autolabel(score_plots[i])

    fig.tight_layout()
    plt.show()
    
    
def plot_line_chart(scores, group_labels, title='Scores for each classifier', bar_width=0.15, labels=g_labels, figure_size=(14, 8)):
    """Plot the contents of scores as a bar graph

    :param scores: A list containing lists of values to plot
    :type scores: list(list(number))
    :param group_labels: A list of labels for each sub-element in scores
    :type group_labels: list(string)
    :param title: Set the title of the figure, defaults to 'Scores for each classifier'
    :type title: str, optional
    :param bar_width: define the width of the bars, defaults to 0.15
    :type bar_width: float, optional
    :param labels: Specify the x-axis labels, defaults to g_labels
    :type labels: list(string), optional
    :param figure_size: size of the figure, defaults to (14, 8)
    :type figure_size: tuple, optional
    """
    num_groups = len(group_labels)
    x = np.arange(len(labels))
    
    fig, ax = plt.subplots(figsize=figure_size)
    score_plots = []
    for i in range(num_groups):
        bar = ax.bar(x + (bar_width * i), scores[i], bar_width, label=group_labels[i])
        score_plots.append(bar)

    ax.set_title(title)
    ax.set_ylabel('% Accuracy')
    ax.set_xticks(x + (bar_width * math.floor((num_groups / 2))))
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()

    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    def autolabel(bar_group):
        """Attach a text label above each bar in *bar_group*, displaying its height."""
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate('{}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    for i in range(num_groups):
        autolabel(score_plots[i])

    fig.tight_layout()
    plt.show()


