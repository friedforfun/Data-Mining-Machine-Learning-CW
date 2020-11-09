from math import ceil
import matplotlib.pyplot as plt
import math
import numpy as np
import networkx as nx
from numpy.random import default_rng

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

def class_to_colour(i):
    classes = {
        0: 'tab:blue',
        1: 'tab:orange',
        2: 'tab:green',
        3: 'tab:red',
        4: 'tab:purple',
        5: 'tab:brown',
        6: 'tab:pink',
        7: 'tab:gray',
        8: 'tab:olive',
        9: 'tab:cyan'
    }
    return classes.get(i)

g_labels = [label_def.get(i) for i in range(-1, 10)]

def show_models(models_list, nrows=2, ncols=6, hide_last=True):
    """Plot the models edges

    :param models_list: model
    :type models_list: list
    :param nrows: Number of rows, defaults to 2
    :type nrows: int, optional
    :param ncols: Number of columns, defaults to 6
    :type ncols: int, optional
    :param hide_last: set visibility of last to false, defaults to True
    :type hide_last: bool, optional
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, 5))
    ax = axes.ravel()
    for i in range(len(models_list)):
        ax[i].axis('off')
        ax[i].set_title(label_def.get(i-1))
        nx.draw(models_list[i][0], with_labels=True, ax=ax[i])

    if hide_last:
        ax[-1].set_visible(False)
    fig.tight_layout()
    plt.show()

def plot_line_graph(data_to_plot, x_label_text='Pixel by rank', y_label_text='Accuracy(%)', title_text='Class by class pixel selection accuracy'):
    """Generate n images from the dataset 

    :param data_to_plot: A list containing the np arrays to plot
    :type data_to_plot: list[np.array]
    :param number_of_lines_to_plot: The number of lines to plot
    :type number_of_lines_to_plot: int, optional
    :param x_label_text: The label of the x axis 
    :type x_label_text: string, optional
    :param y_label_text: The label of the y axis
    :type y_label_text: string, optional
    :param title_text: The title of the graph
    :type title_text: string, optional
    """

    number_of_lines_to_plot = data_to_plot.shape[1]

    line_holder = []

    # The amount of points to plot on the x-axis
    number_of_points = data_to_plot[:,0].shape[0] + 1 

    # Creat the x axis
    x_axis = [i for i in range(1, number_of_points)]

    #add our lines to a linder holder list
    for i in range(data_to_plot.shape[1]):
        line_holder.append(data_to_plot[:,i])

    # Sets the size of the chart
    fig, ax = plt.subplots(figsize=(15, 15))

    # add lines to chart
    # add labels
    for i in range(number_of_lines_to_plot):
        ax.plot(x_axis,line_holder[i],label=label_def[i-1])

    # add legends to the graph
    ax.legend()

    ax.set(xlabel=x_label_text, ylabel=y_label_text, title=title_text)
    ax.grid()
    plt.show()

def plot_images(data, n=5, rows=1, cols=10, figsize=(15, 8), shuffle=True):
    """Generate n images from the dataset 

    :param data: matrix of square images
    :type data: numpy.array
    :param n: number of images to print, defaults to 5
    :type n: int, optional
    """

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    ax = axes.ravel()

    if shuffle:
        rng = default_rng()
        rng.shuffle(data)

    d = int(round(math.sqrt(data.shape[1]), 0))
    if n > data.shape[0]:
        n = data.shape[0]

    for i in range(n):
        row = data[i]
        image = row.reshape(d, d)
        ax[i].imshow(image, cmap='gray')

    for i in range(len(ax)):
        ax[i].axis('off')
        ax[i].set_adjustable('box')

    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    plt.margins(0,0)
    fig.tight_layout()
    plt.show()


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


def plot_scores(scores, group_labels, title='Scores for each classifier', bar_width=0.15, labels=g_labels, figure_size=(14, 8), y_label='% Accuracy'):
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
    ax.set_ylabel(y_label)
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


def scatter_clusters_by_class(kmeans, data_labelled, size=(10, 5), alpha=0.6, legend=(0, 10)):
    """Produce scatter graph for clustering with classes as legend

    :param kmeans: the predicted kmeans labels
    :type kmeans: np.array
    :param data_labelled: data with class labels
    :type data_labelled: pandas.DataFrame
    :param size: size of the figure to plot, defaults to (10, 5)
    :type size: tuple, optional
    :param alpha: the alpha for the plots on the scatter graph, defaults to 0.6
    :type alpha: float, optional
    """

    fig, ax = plt.subplots(figsize=size)
    num_clusters = np.unique(kmeans)
    for i in num_clusters:
        local_cluster = data_labelled[kmeans == i]
        u_labels = np.unique(local_cluster[['y']].to_numpy().flatten())
        for j in u_labels:
            x_ax = local_cluster[local_cluster['y'] == j][0]
            y_ax = local_cluster[local_cluster['y'] == j][1]
            ax.scatter(x_ax, y_ax, color=class_to_colour(
                j), label=j, alpha=alpha)
    ax.legend(range(legend[0], legend[1]))
    ax.set_title('Class true labels')
    fig.tight_layout()
    plt.show()


def scatter_clusters(kmeans, data, labels, size=(10, 5), alpha=1):
    """Plot the clusters on the scatter graph

    :param kmeans: the predicted kmeans labels
    :type kmeans: np.array
    :param data: data with class labels
    :type data: pandas.DataFrame
    :param labels: Dataframe of the true labels for the data
    :type labels: pandas.DataFrame
    :param size: size of the figure to plot, defaults to (10, 5)
    :type size: tuple, optional
    :param alpha: the alpha for the plots on the scatter graph, defaults to 1
    :type alpha: int, optional
    """

    fig, ax = plt.subplots(figsize=size)
    u_labels = np.unique(labels)
    for i in u_labels:
        ax.scatter(data[kmeans == i][0],
                   data[kmeans == i][1], label=i, alpha=alpha)
    ax.legend()
    ax.set_title('Cluster assigned labels')
    fig.tight_layout()
    plt.show()
    
