from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator, HillClimbSearch, ExhaustiveSearch, K2Score, MaximumLikelihoodEstimator, BicScore, BDeuScore
from pgmpy.inference import VariableElimination
import numpy as np
import pandas as pd
import networkx as nx

from Scripts import helperfn as hf
from Scripts import pixelFinder as pf



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
    model = BayesianModel(edges.edges)
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
  
    good_pred = 0
    for i in range(test_data.shape[0]):
        print('looping')
        ev = test_data.iloc[i].to_dict()

        q = model.query(variables=[result_label], evidence=ev)
        pred = np.argmax(q)

        if pred == labels.iloc[i].values[0]:
            good_pred += 1

    score = good_pred / test_data.shape[0]
    print()
    print()
    print('Score: ', score)
    return score
