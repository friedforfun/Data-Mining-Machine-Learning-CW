from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, cohen_kappa_score, plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from .. import helperfn
from itertools import product

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

    #return helperfn.get_results(result_id=0)
    #anomoulous_data = helperfn.get_results(result_id=-1)
    #print('Dataset: ', -1, ' Has results:', np.unique(anomoulous_data.to_numpy()))

    #raw_data_results = raw_data_results + [anomoulous_data]

    for i in range(0, 11):
        if balance_classes:
            raw_y = helperfn.get_results(result_id=i-1)
            raw_data_results += [helperfn.balance_by_class(training_smpl, raw_y)]
            print('Dataset: ', i-1, ' Has results:',
                  np.unique(raw_data_results[i][1].to_numpy()))
            #print(raw_data_results[i][1])
            train_test_data = train_test_data + [train_test_split(raw_data_results[i][0], raw_data_results[i][1], test_size=test_size, random_state=random_state)]

        else:
            raw_data_results = raw_data_results + [helperfn.get_results(result_id=i-1)]
            print('Dataset: ', i-1, ' Has results:', np.unique(raw_data_results[i].to_numpy()))
            train_test_data = train_test_data + [train_test_split(training_smpl, raw_data_results[i], test_size=test_size, random_state=random_state)]
        
        print(train_test_data[i][0].shape, train_test_data[i][1].shape, train_test_data[i][2].shape, train_test_data[i][3].shape)
        print(train_test_data[i][1].values.dtype)
        classifiers = classifiers + \
            [GaussianNB().fit(train_test_data[i][0].values,
                              train_test_data[i][2].values)]
        scores = scores + [(classifiers[i].score(train_test_data[i][0], train_test_data[i][2]),
                            classifiers[i].score(train_test_data[i][1], train_test_data[i][3]))]

    for i in range(len(scores)):
        print("Scores for dataset: ", label_def.get(i-1, i-1))
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
    train_confusion = []
    confusion = []
    # build confusion matrices for all classifiers
    for i in range(len(classifiers)):
        cmt = confusion_matrix(data[i][2], classifiers[i].predict(data[i][0]))
        cm = confusion_matrix(data[i][3], classifiers[i].predict(data[i][1]))
        confusion = confusion + [cm]
        train_confusion += [cmt]
    return train_confusion, confusion

def show_confusion_matrix(confusion, index_range=(0, 10), kappas=None):
    """Diplay all confusion matrix within range

    :param confusion: the list of confusion matrices
    :type confusion: List: np.ndarray
    :param index_range: the range of indices within the confusion param to display, defaults to (0, 10)
    :type index_range: tuple(int), optional
    """
    fig, axes = plt.subplots(nrows=len(confusion), figsize=(7, len(confusion)*10))
    ax = axes.ravel()
    for i in range(index_range[0], index_range[1]):
        if kappas is not None:
            #print('Kappa: ', kappas[i])
            kappa = round(kappas[i], 2)
        if i == 0:
            # special case labels
            cmd = ConfusionMatrixDisplay(confusion[i], display_labels=[i for i in range(0,10)])
            cmd.plot(ax=ax[i])
            if kappas is not None:
                cmd.ax_.set_title(f'All classes\nKappa: {kappa}', fontsize=20)
            else:
                cmd.ax_.set_title('All classes', fontsize=20)
        else:
            cmd = ConfusionMatrixDisplay(
                confusion[i], display_labels=['True', 'False'])
            cmd.plot(ax=ax[i])
            label = label_def.get(i, i)
            if kappas is not None:
                cmd.ax_.set_title(f'{label}\n Kappa: {kappa}', fontsize=20)
            else:
                cmd.ax_.set_title(f'{label}', fontsize=20)
            
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    axes.flat[-1].set_visible(False)
    plt.show()

def multi_show_confusion_matrix(conf_arr, index_range=(0,10), kappas=None, col_labels=None):
    """Display all confusion matrices together

    :param conf_arr: List containing arrays of confusion matrices
    :type conf_arr: List[List[numpy.array]]
    :param index_range: the range of indices within the confusion param to display, defaults to (0, 10), defaults to (0,10)
    :type index_range: tuple(int), optional
    :param kappas: List of Kappa values matching the shape of conf_arr, defaults to None
    :type kappas: List[List[numpy.array]], optional
    :param col_labels: Description of each dataset
    :type col_labels: List[string]
    """
    if kappas is not None:
        if len(conf_arr) != len(kappas):
            raise ValueError('mismatched conf_arr and kappas params')

    # all i in conf_arr[i] must have equal length
    fig, axes = plt.subplots(nrows=len(conf_arr[0]), ncols=len(conf_arr), figsize=(len(conf_arr)*10, len(conf_arr[0])*10))
    ax_arr = axes.ravel()

    for i in range(len(conf_arr)):
        for j in range(len(conf_arr[i])):
            if j == 0:
                disp_labels = [i for i in range(0, 10)]
                title = f'{col_labels[i]}\nAll classes'
                if kappas is not None:
                    kappa = round(kappas[i][j], 2)
                    title = f'{col_labels[i]}\nAll classes\nKappa: {kappa}'
            else:
                disp_labels = ['True', 'False']
                label = label_def.get(j-1, j-1)
                title = label
                if kappas is not None:
                    kappa = round(kappas[i][j], 2)
                    title = f'{label}\nKappa: {kappa}'
            cmd = ConfusionMatrixDisplay(conf_arr[i][j], display_labels=disp_labels)
            cmd.plot(ax=ax_arr[j*len(conf_arr)+i])
            cmd.ax_.set_title(title, fontsize=20)


    plt.tight_layout(h_pad=5, w_pad=2)
    
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
