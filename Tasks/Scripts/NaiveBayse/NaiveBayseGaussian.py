from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from IPython.utils import io
from .. import helperfn
from .. import downsample as ds


def nbg_model_custom_data(X, y, data_label=None, test_size=0.2, random_state=0, balance_classes=False, print_scores=True):
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
    :rtype: (GaussianNB, (training_scores, testing_scores), (sklearn.train_test_split))
    """
    
    if balance_classes:
        X = pd.DataFrame(data=X)
        y = pd.DataFrame(data=y)
        X, y = helperfn.balance_by_class(X, y)
        X = X.astype(int)
        y = y.astype(int)
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    classifier = GaussianNB().fit(X_train, y_train)
    score_train = classifier.score(X_train, y_train)
    score_test = classifier.score(X_test, y_test)

    if print_scores:
        if data_label is not None:
            print("Scores for dataset: ", label_def.get(data_label, data_label))
        print("Training data score: ", score_train)
        print("Testing data score: ", score_test)
        print("--------------------------------------")

    return classifier, (score_train, score_test), (X_train, X_test, y_train, y_test)


def build_nbg_models(downscale=False, downscale_shape=(2, 2), print_scores=True,**kwargs):
    """Build and score naive bayse gaussian model

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
    train_test_data = []
    classifiers = []
    scores = []

    for i in range(0, 11):
        results = helperfn.get_results(result_id=i-1)
        if print_scores:
            print('Dataset: ', i-1, ' Has results:',np.unique(results.to_numpy()))
        classifer, score, data = nbg_model_custom_data(
            training_smpl, results, **kwargs, data_label=i-1)
        classifiers += [classifer]
        scores += [score]
        train_test_data += [data]

    return classifiers, scores, train_test_data


def run_classifier(x_data, y, pixel_order, n_pixels=5, verbose=False, **kwargs):
    """[summary]

    :param x_data: [description]
    :type x_data: [type]
    :param y: [description]
    :type y: [type]
    :param pixel_order: [description]
    :type pixel_order: [type]
    :param n_pixels: [description], defaults to 5
    :type n_pixels: int, optional
    :param verbose: [description], defaults to False
    :type verbose: bool, optional
    :return: [description]
    :rtype: [type]
    """
    scores_list = []
    classifiers_list = []
    data_list = []

    #pixels = grab_n_pixels(pixel_order, 0)

    # No 0 pixels so start at 1 
    for i in range(1, n_pixels + 1):
        helperfn.update_progress(i/(n_pixels), message='running all classifiers, could be slow')
        pixels = helperfn.grab_n_pixels(pixel_order, i)
        #print(len(pixels))
        if not verbose:
            print('Classifying pixel: ' , i)
            with io.capture_output() as captured:
                classifier, scores, data = build_classifiers(x_data, y, pixels, **kwargs)
        else:
            classifier,  scores, data = build_classifiers(x_data, y, pixels, **kwargs)
        
        scores_list.append(scores)
        data_list.append(data)
        classifiers_list.append(classifier)
    
    return scores_list #, data_list, classifiers_list

# This range could be incorrect might need to be (0,12)
def build_classifiers(data, y_labels, pixel_order, result_label_set=(0,11), **kwargs):
    """[summary]

    :param data: [description]
    :type data: [type]
    :param y_labels: [description]
    :type y_labels: [type]
    :param pixel_order: [description]
    :type pixel_order: [type]
    :param result_label_set: [description], defaults to (0,11)
    :type result_label_set: tuple, optional
    :return: [description]
    :rtype: [type]
    """
    
    classifiers = []
    scores = []
    dataset = []
    for i in range(result_label_set[0], result_label_set[1]):
        X = np.take(data, pixel_order[i], axis=1)
        if X.shape[1] == 0:
            X = np.take(data, [pixel_order[i]], axis=1)
            #print(X)
        #print('THIS IS X', X.shape)
        y = y_labels[i]
        classifier, score, local_data = nbg_model_custom_data(X, y, data_label=i-1, **kwargs)
        classifiers += [classifier]
        scores += [score]
        dataset += [local_data]

    return classifiers, scores, dataset

def using_n_pixelrun_classifier(x_data, y, pixel_order,best_pixel_indicies, n_pixels=5, verbose=False, **kwargs):
    """[summary]

    :param x_data: [description]
    :type x_data: [type]
    :param y: [description]
    :type y: [type]
    :param pixel_order: [description]
    :type pixel_order: [type]
    :param best_pixel_indicies: [description]
    :type best_pixel_indicies: [type]
    :param n_pixels: [description], defaults to 5
    :type n_pixels: int, optional
    :param verbose: [description], defaults to False
    :type verbose: bool, optional
    :return: [description]
    :rtype: [type]
    """
    
    scores_list = []
    classifiers_list = []
    data_list = []

    #pixels = grab_n_pixels(pixel_order, 0)

    # No 0 pixels so start at 1 
    for i in range(0, 11):
        #hf.update_progress(i/10, message='building all classifiers with best pixel amount for each')
        pixels = helperfn.grab_n_pixels(pixel_order, best_pixel_indicies[i])

        if not verbose:
            print('Classifying class: ' , i)
            with io.capture_output() as captured:
                classifier, scores, data = build_classifiers(x_data, y, pixels,result_label_set=(i,i+1), **kwargs)
        else:
            classifier,  scores, data = build_classifiers(x_data, y, pixels, result_label_set=(i,i+1), **kwargs)
        
        scores_list.append(scores)
        data_list.append(data)
        classifiers_list.append(classifier)
    
    return classifiers_list, scores_list, data_list


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
