import numpy as np
from . import helperfn as hf
import matplotlib.pyplot as plt


def bestPixels(label, n):
    """Get n number of best pixels indexes from dataset label

    :param label: label_def as defined in helperfn
    :type label: int
    :return: Array of size n with idexes of n best pixels
    :rtype: array
    """
    X, y = hf.get_data(label)

    result = []
    for i in range(X.shape[1]):
        result.append(X[str(i)].corr(y['y']))

    result = np.array(result)

    res_arr = []

    for i in range(n):

        index = np.argmax(result)
        res_arr.append(index)
        result[index] = -111

    return res_arr

def displayBest(n, index_range=(0, 10)):
    """Display n best pixels indexes from datasets defined in index_range

    :param index_range: Range of datasets
    :type index_range: int tuple
    :param label: label_def as defined in helperfn
    :type label: int
    """
    for i in range(index_range[0], index_range[1]):
        print(str(n) + " best pixels from dataset " + str(i) + " " + bestPixels(i, n))


def showHeatmap(index_range=(0, 10)):
    """Display heatmaps from datasets defined in index_range

    :param index_range: Range of datasets
    :type index_range: int tuple
    """
    for i in range(index_range[0], index_range[1]):

        sort = np.empty(2304, dtype=int)
        bestPix = bestPixels(i, 2304)

        for j in range(2304):
            index = bestPix[j]
            sort[index] = j


        sort2d = np.reshape(sort, (-1, 48))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(sort2d, interpolation='nearest', cmap='inferno')
        plt.title(hf.result_files[i])
        fig.colorbar(cax)

        plt.show()

