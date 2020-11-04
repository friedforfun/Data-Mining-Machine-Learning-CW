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
    return np.array([i for i, j in score]), np.array([j for i, j in score])

def convert_percentage(score_tuple):
    return score_tuple[0] * 100, score_tuple[1] * 100


def plot_scores(scores, group_labels, title='Scores for each classifier', bar_width=0.15, labels=g_labels, figure_size=(14, 8)):

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


