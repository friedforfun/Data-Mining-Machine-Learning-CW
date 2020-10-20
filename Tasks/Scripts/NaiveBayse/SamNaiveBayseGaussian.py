from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, cohen_kappa_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.utils import indices_to_mask
from .. import helperfn
from itertools import product
from sklearn.model_selection import RepeatedStratifiedKFold

import numpy as np

def build_nbg_models(test_size=0.2, random_state=0, balance_classes=False):
    """Build and score naive bayse gaussian model

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
    
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,random_state=36851234)

    #return helperfn.get_results(result_id=0)
    #anomoulous_data = helperfn.get_results(result_id=-1)
    #print('Dataset: ', -1, ' Has results:', np.unique(anomoulous_data.to_numpy()))

    #raw_data_results = raw_data_results + [anomoulous_data]

    for i in range(0, 11):
        if balance_classes:
            raw_y = helperfn.get_results(result_id=i-1)
            #raw_data_results += [helperfn.balance_by_class(training_smpl, raw_y)]
            print('Dataset: ', i-1, ' Has results:',
                  np.unique(raw_data_results[i][1].to_numpy()))
            train_test_data = train_test_data + [train_test_split(raw_data_results[i][0], raw_data_results[i][1], test_size=test_size, random_state=random_state)]

        else:
            raw_data_results = raw_data_results + [helperfn.get_results(result_id=i-1)]
            print('Dataset: ', i-1, ' Has results:', np.unique(raw_data_results[i].to_numpy()))
            train_test_data = train_test_data + [train_test_split(training_smpl, raw_data_results[i], test_size=test_size, random_state=random_state)]
            
        
        
        for train_index, test_index in rskf.split(train_test_data[i][0], train_test_data[i][1]):
            X_train, X_test = train_test_data[train_index][0], train_test_data[test_index][0]
            y_train, y_test = train_test_data[train_index][1], train_test_data[test_index][1]
        
            print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
            print(X_test.values.dtype)
            classifiers = classifiers + \
                [GaussianNB().fit(X_train.values,
                                  y_train.values)]
            scores = scores + [(classifiers[i].score( X_train, y_train),
                                classifiers[i].score(X_test, y_test))]

    for i in range(len(scores)):
        print("Scores for dataset: ", label_def.get(i-1, i-1))
        print("Training data score: ", scores[i][0])
        print("Testing data score: ", scores[i][1])
        print("--------------------------------------")

    return classifiers, scores, train_test_data


def rskSplit():
    
    from sklearn.model_selection import RepeatedStratifiedKFold
    
    classifiers = []
    scores = []
    X , y = helperfn.get_data(2)

    
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,random_state=36851234)
        
    for train_index, test_index in rskf.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        print(train_index.shape , test_index.shape)
        X_train, X_test = np.take(X,train_index, axis=0), np.take(X,test_index, axis=0)
        y_train, y_test = np.take(y,train_index, axis=0), np.take(y,test_index, axis=0)
        
        
    for i in range(0, 11):
    
        classifiers = classifiers + \
                    [GaussianNB().fit(X_train.values,
                                    y_train.values)]
        scores = scores + [(classifiers[i].score( X_train, y_train),
                                    classifiers[i].score(X_test, y_test))]

        for i in range(len(scores)):
            print("Scores for dataset: ", label_def.get(i-1, i-1))
            print("Training data score: ", scores[i][0])
            print("Testing data score: ", scores[i][1])
            print("--------------------------------------")
        


def build_confusion_matrix(classifiers, data):
    """Build the confusion matrices

    :param classifiers: a list of all the classifiers to generate confusion matrices for
    :type classifiers: List: sklearn.naive_bayes.CategoricalNB
    :param data: List of the Train and testing data
    :return: List: sklearn.metrics.confusion_matrix
    """
    confusion = []
    # build confusion matrices for all classifiers
    for i in range(len(classifiers)):
        cm = confusion_matrix(data[i][3], classifiers[i].predict(data[i][1]))
        confusion = confusion + [cm]
    return confusion

def show_confusion_matrix(confusion, index_range=(0, 10), kappas=None):
    """Diplay all confusion matrix within range

    :param confusion: the list of confusion matrices
    :type confusion: List: np.ndarray
    :param index_range: the range of indices within the confusion param to display, defaults to (0, 10)
    :type index_range: tuple(int), optional
    """
    for i in range(index_range[0], index_range[1]):
        if kappas is not None:
            print('Kappa: ', kappas[i])
        fig, ax = plt.subplots()
        cax = ax.matshow(confusion[i])
        if i == 0:
            # special case labels
            cmd = ConfusionMatrixDisplay(confusion[i], display_labels=[i for i in range(0,10)])
            plt.title("All classes")
        else:
            cmd = ConfusionMatrixDisplay(
                confusion[i], display_labels=['True', 'False'])
            plt.title(label_def.get(i, i))

        cmd.plot(ax=ax)
        plt.show()
        

def kappa(confusion):
    """Generate kappa values for the confusion matrix against randomised ndarrays

    :param confusion: the list of confusion matrices
    :type confusion: List: np.ndarray
    :return: List of kappa values between -1 and 1, a positive kappa values indicates better than chance
    :rtype: List[float]
    """
    res = []
    for i in confusion:
        _pred_totals = np.sum(i, axis=0) # sum into row axis, but its represented by columns
        _true_totals = np.sum(i, axis=1) # represented by rows
        _sum = np.sum(i)
        _norm_i = i / _sum
        p_observed = 0
        p_chance = 0
        _chance_matrix = np.zeros(i.shape)
        for j in range(i.shape[0]):
            for k in range(i.shape[1]):
                _chance_matrix[j][k] = _pred_totals[k] / _sum * _true_totals[j]/_sum
            p_observed += _norm_i[j][j]
            p_chance += _chance_matrix[j][j]
        res.append(_calc_kappa(p_observed, p_chance))
        #res.append(cohen_kappa_score(i, _chance_matrix))
    return res

def _calc_kappa(p_observed, p_chance):
    return (p_observed - p_chance) / (1 - p_chance)


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
