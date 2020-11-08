from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator, HillClimbSearch, ExhaustiveSearch, K2Score, MaximumLikelihoodEstimator, BicScore, BDeuScore
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from IPython.utils import io
import numpy as np
import pandas as pd
import networkx as nx

from . import pixelFinder as pf
from . import helperfn as hf
from . import downsample as ds


def build_networks(best_pixel_labels, balance_by_class=False, ewb=True, downscale=False, result_label_set=(0, 11), **kwargs):
    """Build the networks for all specified result label sets

    :param best_pixel_labels: An object representing all the pixels to use as nodes in the bayse network
    :type best_pixel_labels: List
    :param balance_by_class: Balance the class distributions for each class, defaults to False
    :type balance_by_class: bool, optional
    :param ewb: Perform Equal width binning on the data, defaults to True
    :type ewb: bool, optional
    :param downscale: Downscale the data, defaults to False
    :type downscale: bool, optional
    :return: Tuple of model, scores, and data
    :rtype: (list, list, list)
    """
    X, y = hf.get_data()

    if downscale:
        X = ds.downscale(X)

    if ewb:
        X = hf.to_ewb(X)

    models_inference = []
    scores = []
    train_test_data = []

    for i in range(result_label_set[0], result_label_set[1]):
        hf.update_progress(i / result_label_set[1]-1,
                           message='Building all networks...')
        y = hf.get_results(result_id=i-1)
        X_ = X
        if balance_by_class:
            X_, y = hf.balance_by_class(X, y)

        X_ = np.take(X_, best_pixel_labels[i], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X_, y, **kwargs)

        X_ = X_train.join(y_train)

        edge_estimator = estimate_model_edges(X_, **kwargs)
        model = model_with_params(
            X_, edge_estimator.edges, **kwargs)

        infer = get_inference_model(model)

        train_score = score_model(infer, X_train, y_train)
        test_score = score_model(infer, X_test, y_test)
        
        models_inference.append([model, infer])
        scores.append([train_score, test_score])
        train_test_data.append([X_train, X_test,  y_train, y_test])

    return models_inference, scores, train_test_data

def get_naive_edges(pixels, label='y'):
    """ Naively assume that the label attribute is the only dependancy

    :param pixels: List of nodes
    :type pixels: list(str)
    :return: a list of tuples representing all the edges in the graph
    :rtype: list(str, str)
    """
    edges = [(i, label) for i in list(map(str, pixels))]
    return edges

def estimate_model_edges(data, scorer='k2', learning_algo='HillClimb', max_indegree=4, max_iter=int(1e4), equivalent_sample_size=8):
    """Estimate the edges of the model given some training data (with labels)

    :param data: Labelled training dataframe
    :type data: Pandas.DataFrame
    :param scorer: The scoring function, defaults to 'k2'
    :type scorer: str, optional
    :param learning_algo: The learning algorithm, defaults to 'HillClimb'
    :type learning_algo: str, optional
    :param max_indegree:  If provided only search among models where all nodes have at most max_indegree parents, defaults to 4
    :type max_indegree: int, optional
    :param max_iter: The maximum number of iterations allowed. Returns the learned model when the number of iterations is greater than max_iter, defaults to int(1e4)
    :type max_iter: int, optional
    :param equivalent_sample_size: The equivalent/imaginary sample size (of uniform pseudo samples) for the dirichlet hyperparameters.
            The score is sensitive to this value, runs with different values might be useful, defaults to 8
    :type equivalent_sample_size: int, optional
    :raises ValueError: "Scorer should be one of: K2, Bic, or BDeu"
    :raises ValueError: "learning algorithm should be one of: HillClimb or ExhaustiveSearch"
    :return: A model at a local score maximum
    :rtype: DAG instance
    """
    
    if scorer.upper() == 'k2'.upper():
        score = K2Score(data=data)
    elif scorer.upper() == 'bic'.upper():
        score = BicScore(data=data)
    elif scorer.upper() == 'BDeu'.upper():
        score = BDeuScore(
            data=data, equivalent_sample_size=equivalent_sample_size)
    else:
        raise ValueError("Scorer should be one of: K2, Bic, or BDeu")

    if learning_algo.upper() == 'HillClimb'.upper():
        est = HillClimbSearch(data=data, scoring_method=score)
    elif learning_algo.upper() == 'ExhaustiveSearch'.upper():
        est = ExhaustiveSearch(data=data, scoring_method=score)
    else:
        raise ValueError("learning algorithm should be one of: HillClimb or ExhaustiveSearch")

    print("---- Beginning edge estimator ----")
    print("WARNING: this might be very slow")

    estimated_edges = est.estimate(max_indegree=max_indegree, max_iter=max_iter)

    print('---- Done ----')
    return estimated_edges

# estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=(int)
# estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=(int)
# estimator=BayesianEstimator, prior_type="K2"
def model_with_params(data, edges, estimator=MaximumLikelihoodEstimator, **kwargs):
    """Build the model and fit the parameters

    :param data: Labelled training dataframe
    :type data: Pandas.DataFrame
    :param edges: The edges of a model
    :type edges: List of edge tuples
    :param estimator: Either MaximumLikelihoodEstimator or BayesianEstimator, defaults to MaximumLikelihoodEstimator
    :type estimator: pgmpy.estimator, optional
    :return: the model with learned parameters
    :rtype: DAG instance
    """
    model = BayesianModel(edges)
    print(model.nodes)
    model.fit(data=data, estimator=estimator, **kwargs)
    return model

def get_inference_model(model):
    model.check_model()
    return VariableElimination(model)

def score_model(model, test_data, labels, result_label='y'):
    """Produce a score for the model by testing all the test_data against the labels

    :param model: A model with edges and parameters defined
    :type model: DAG model
    :param test_data: unlabeled testing data
    :type test_data: pandas.DataFrame
    :param labels: the labels for the test data
    :type labels: pandas.DataFrame
    :param result_label: The variable label in the model being tested, defaults to 'y'
    :type result_label: str, optional
    :return: The percentage of good predictions
    :rtype: float
    """
  
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    predictions = []

    for i in range(test_data.shape[0]):
        ev = test_data.iloc[i].to_dict()
        hf.update_progress(i / test_data.shape[0], message='Scoring model...')
        with io.capture_output() as captured:
            q = model.map_query(variables=[result_label], evidence=ev)
        
        pred = q.get(result_label)
        predictions.append(pred)

        if labels.iloc[i].values[0] == 0 and pred == 0:
            true_positive += 1
        if labels.iloc[i].values[0] == 1 and pred == 1:
            true_negative += 1
        if labels.iloc[i].values[0] == 0 and pred == 1:
            false_negative += 1
        if labels.iloc[i].values[0] == 1 and pred == 0:
            false_positive += 1

    score = (true_positive+true_negative) / test_data.shape[0]
    print()
    print()
    print('Score: ', score)
    return ((true_positive, false_positive, false_negative, true_negative), predictions)

def bayse_net_confusion_matrices(scores, data):
    conf_train = []
    conf_test = []
    for i in range(len(scores)):
        train_pred = scores[i][0][1]
        train_labels = data[i][2].to_numpy().flatten().tolist()
        test_pred = scores[i][1][1]
        test_labels = data[i][3].to_numpy().flatten().tolist()
        conf_train.append(confusion_matrix(train_labels, train_pred))
        conf_test.append(confusion_matrix(test_labels, test_pred))
    return conf_train, conf_test
