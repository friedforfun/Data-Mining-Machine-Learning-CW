import numpy as np
from . import helperfn as hf
# from NaiveBayse import SamNaiveBayseGaussian as nbg
import matplotlib.pyplot as plt


def bestPixels(label, n, downscale=False):
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


def get_top_pixels(n):
    """Get the top n pixels for all datasets

    :param n: The number of pixels to get
    :type n: int
    :return: a list containing 11 lists of column indices, each for the different classification
    :rtype: List[List[int]]
    """
    pixel_order = []
    for i in range(-1, 10):
        pixel_order.append(np.array(bestPixels(i, n)))
    return pixel_order


def data_lists():
    """ Lists of the dataset and the result column

    :return: a list containing 11 lists, each containing X and y data
    :rtype: List
    """
    data = []
    for i in range(-1, 10):
        data.append(hf.get_data(i))
    return data


# def build_classifiers(data, pixel_order, balance_classes=False):
#     """ Build classifiers based on the pixel order, for 11 datasets

#     :param data: the data to use to build the classifier
#     :type data: np.array
#     :param pixel_order: A list of pixels by priority
#     :type pixel_order: List[int]
#     :param balance_classes: Balance the class distribution or not, defaults to False
#     :type balance_classes: bool, optional
#     :return: The classifier, scores and dataset for these parameters
#     :rtype: Tuple(List[classifer], List[scores], List[datasets])
#     """
#     classifiers = []
#     scores = []
#     dataset = []
#     for i in range(1, 11):
#         X = np.take(data[i][0], pixel_order[i], axis=1)
#         classifier, score, local_data = nbg.nbg_model_custom_data(
#             X, y, data_label=i-1, balance_classes=balance_classes)
#         classifiers += [classifier]
#         scores += [scores]
#         dataset += [local_data]

#     return classifiers, scores, dataset
