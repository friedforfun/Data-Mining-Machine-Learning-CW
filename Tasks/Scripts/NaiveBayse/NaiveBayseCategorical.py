from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
import pandas as pd
from .. import helperfn
from .. import downsample as ds

import numpy as np


def nbc_model_custom_data(X, y, data_label=None, test_size=0.2, random_state=0, balance_classes=False, print_scores=True, size=None, allow_imbalance=False):
    """Build classifiers, scores, and data from supplied dataset

    :param X: The data
    :type X: np.array
    :param y: The data labels
    :type y: np.array
    :param data_label: An integer identifier for the labels, defaults to None
    :type data_label: int, optional
    :param test_size: the percentage of the sample size to test with, defaults to 0.2
    :type test_size: float, optional
    :param random_state: the random seed, defaults to 0
    :type random_state: int, optional
    :param balance_classes: Set to true to create a balanced class distribution, defaults to False
    :type balance_classes: bool, optional
    :param print_scores: display the scores, defaults to True
    :type print_scores: bool, optional
    :return: The classifier, scores, and data
    :rtype: (CategoricalNB, (training_scores, testing_scores), (sklearn.train_test_split))
    """
    

    if balance_classes:
        X = pd.DataFrame(data=X)
        y = pd.DataFrame(data=y)
        X, y = helperfn.balance_by_class(
            X, y, size=size, allow_imbalance=allow_imbalance, random_state=random_state)
        X = X.astype(int)
        y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    classifier = CategoricalNB()
    classifier = classifier.fit(X_train, y_train)
    score_train = classifier.score(X_train, y_train)
    score_test = classifier.score(X_test, y_test)

    if print_scores:
        if data_label is not None:
            print("Scores for dataset: ", label_def.get(data_label, data_label))
        print("Training data score: ", score_train)
        print("Testing data score: ", score_test)
        print("--------------------------------------")

    return classifier, (score_train, score_test), (X_train, X_test, y_train, y_test)


def build_nbc_models(downscale=False, downscale_shape=(2,2), ewb=False, **kwargs):
    """Build and score naive bayse categorical model

    :param downscale: Downscale the images by a factor defined in downscale_shape param
    :type downscale: bool, optional
    :param downscale_shape: The degree of downscaling on each axis, defaults to (2,2)
    :type downscale_shape: tuple, optional
    :return: Tuple of List: classifers, List: scores, List: traing & testing data
    :rtype: Tuple(List, List, List)
    """
    training_smpl = helperfn.get_data_noresults()
    if downscale:
        training_smpl = ds.downscale(training_smpl, downscale_shape=downscale_shape)
    if ewb:
        training_smpl = helperfn.to_ewb(pd.DataFrame(training_smpl))

    train_test_data = []
    classifiers = []
    scores = []


    for i in range(0, 11):
        results = helperfn.get_results(result_id=i-1)
        print('Dataset: ', i-1, ' Has results:',np.unique(results.to_numpy()))
        classifer, score, data = nbc_model_custom_data(
            training_smpl, results, **kwargs, data_label=i-1)
        classifiers += [classifer]
        scores += [score]
        train_test_data += [data]

    return classifiers, scores, train_test_data

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
