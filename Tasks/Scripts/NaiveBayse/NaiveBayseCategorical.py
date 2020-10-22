from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.naive_bayes import CategoricalNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from .. import helperfn

import numpy as np

def build_nbc_models(test_size=0.2, random_state=0):
    """Build and score naive bayse categorical model

    :param test_size: the percentage of the sample size to test with, defaults to 0.2
    :type test_size: float, optional
    :param random_state: the random seed, defaults to 0
    :type random_state: int, optional
    :return: Tuple of List: classifers, List: scores, List: traing & testing data
    :rtype: Tuple(List, List, List)
    """
    training_smpl = helperfn.get_data_noresults()
    raw_data_results = []

    train_test_data = []
    classifiers = []
    scores = []

    for i in range(-1, 10):
        raw_data_results = raw_data_results + [helperfn.get_results(result_id=i)]
        print('Dataset: ', i, ' Has results:', np.unique(raw_data_results[i].to_numpy()))
        train_test_data = train_test_data + [train_test_split(training_smpl, raw_data_results[i], test_size=test_size, random_state=random_state)]
        classifiers = classifiers + [CategoricalNB().fit(train_test_data[i][0], train_test_data[i][2])]
        scores = scores + [(classifiers[i].score(train_test_data[i][0], train_test_data[i][2]),
                            classifiers[i].score(train_test_data[i][1], train_test_data[i][3]))]

    for i in range(len(scores)):
        print("Scores for dataset: ", i-1)
        print("Training data score: ", scores[i][0])
        print("Testing data score: ", scores[i][1])
        print("--------------------------------------")

    return classifiers, scores, train_test_data

def build_confusion_matrix(classifiers, data):
    """Build the confusion matrices

    :param classifiers: a list of all the classifiers to generate confusion matrices for
    :type classifiers: List: sklearn.naive_bayes.CategoricalNB
    :param data: List of the Train and testing data
    :return: List: sklearn.metrics.confusion_matrix
    """
    test_confusion = []
    train_confusion = []
    # build confusion matrices for all classifiers
    for i in range(len(classifiers)):
        cmt = confusion_matrix(data[i][2], classifiers[i].predict(data[i][0]))
        train_confusion += [cmt]
        cm = confusion_matrix(data[i][3], classifiers[i].predict(data[i][1]))
        test_confusion = test_confusion + [cm]
    return train_confusion, test_confusion

def show_confusion_matrix(confusion, index_range=(0, 11)):
    for i in range(index_range[0],index_range[1]):
        if i == 0:
            # special case labels
            cmd = ConfusionMatrixDisplay(
                confusion[i], display_labels=['20', '30', '50', '60', '70', 'left', 'right', 'ped Xing', 'beware childer', 'cycle route'])
        else:
            cmd = ConfusionMatrixDisplay(confusion[i], display_labels=['yes', 'no'])
        cmd.plot()

# Silly feature selection
def feature_sel(test_size=0.9, random_state=0):
    training_smpl = helperfn.get_data_noresults()
    raw_data_results = helperfn.get_results(result_id=8)
    X_train, X_test, y_train, y_test = train_test_split(
        training_smpl, raw_data_results, test_size=test_size, random_state=random_state)
    selector = RFE(MultinomialNB(), n_features_to_select=10, step=1)
    selector.fit(X_train, y_train)
    return selector, X_test, y_test
