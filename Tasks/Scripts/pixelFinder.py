import numpy as np

def bestPixels(X, y, n):
    """Get n number of best pixels indexes

    :param X: Dataframe of features
    :type X: dataframe
    :param y: Dataframe of outcome
    :type y: dataframe
    :return: Array of size n with idexes of n best pixels
    :rtype: array
    """
    result = []

    for i in range(X.shape[1]):
        result.append(X[str(i)].corr(y['y']))

    result = np.array(result)


    res_arr = []

    for i in range(n):

        index = np.argmax(result)
        res_arr.append(index)

        result = np.delete(result, index)


    return res_arr


