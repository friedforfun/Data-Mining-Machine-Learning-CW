import numpy as np
from . import helperfn as hf
from . import downsample as ds
# from NaiveBayse import SamNaiveBayseGaussian as nbg
import matplotlib.pyplot as plt


def bestPixels(label, n, downscale=False, downscale_shape=(2, 2)):
    """Fetch data and get n number of best pixels indexes from dataset label

    :param label: label_def as defined in helperfn
    :type label: int
    :return: Array of size n with idexes of n best pixels
    :rtype: array
    """
    X, y = hf.get_data(label)
    if downscale:
        X = ds.downscale(X, downscale_shape=downscale_shape)

    result = []
    for i in range(X.shape[1]):
        result.append(X[str(i)].corr(y['y']))

    result = np.array(result)

    res_arr = []

    for i in range(n):

        index = np.argmax(result)
        res_arr.append(index)
        result[index] = -9223372036854775806

    return res_arr


def grab_n_pixels(pixel_order, n):
    """Select a subset of pixel indices from the pixel_order object

    :param pixel_order: list of pixel names in order of correlation to a given label
    :type pixel_order: list[list[str]]
    :param n: number of pixels to extract
    :type n: int
    :return: sublist of pixel names in order of correlation to a given label
    :rtype: list[list[str]]
    """
    output = []

    # j is the pixel order list
    for j in range(len(pixel_order)):
        output.append(pixel_order[j][:n])

    return output

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


def get_top_pixels(n, downscaled=False):
    """Get the top n pixels for all datasets

    :param n: The number of pixels to get
    :type n: int
    :return: a list containing 11 lists of column indices, each for the different classification
    :rtype: List[List[int]]
    """
    X = hf.get_data_noresults()
    if downscaled:
        X = ds.downscale(X)
    pixel_order = []
    for i in range(-1, 10):
        hf.update_progress(i+1 / 11)
        y = hf.get_results(i)
        y.columns = ['y']
        pixel_order.append(np.array(best_pixels(X, y, n)))
    return pixel_order


def best_pixels(X, y, n, downscale=False, downscale_shape=(2, 2)):
    """Get n number of best pixels indexes from dataset label

    :param X: the training data to use
    :type X: pandas.DataFrame
    :param y: The labels to use
    :type y: pandas.DataFrame
    :param n: The number of pixels to search for
    :type n: int
    :param downscale: Downscale the data?
    :type downscale: bool, optional
    :param downscale_shape: The degree of downscaling on each axis, defaults to (2,2)
    :type downscale_shape: tuple, optional

    :return: List of size n with idexes of n best pixels
    :rtype: List
    """
    if downscale:
        X = ds.downscale(X, downscale_shape=downscale_shape)

    result = []
    for i in range(X.shape[1]):
        result.append(X[str(i)].corr(y['y']))

    result = np.array(result)
    res_arr = []

    for i in range(n):
        index = np.argmax(result)
        res_arr.append(index)
        result[index] = -9223372036854775806

    return res_arr

def get_best_n_pixels_all_classes(data):
    """[summary]

    :param data: [description]
    :type data: [type]
    :return: [description]
    :rtype: [type]
    """
    max_value = []

    data = data.T

    for i in range(data.shape[0]):
        max_value.append(np.argmax(data[i]))
    
    return max_value







