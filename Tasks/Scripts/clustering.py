from collections import Counter, defaultdict
import numpy as np

def ClusterIndicesNumpy(clustNum, labels_array):
    """Get the indices for a cluster and class

    :param clustNum: the cluster number
    :type clustNum: int
    :param labels_array: a dataframe of labels
    :type labels_array: pandas.DataFrame
    :return: list of indices where the label corresponds to a class
    :rtype: np.array
    """
    return np.where(labels_array == clustNum)[0]

def check_shape(cluster):
    """Check the shape of 2 cluster classifications (triangular or circle)

    :param cluster: Cluster dict
    :type cluster: dict
    :return: number of data split by high level features
    :rtype: numpy.array
    """
    sign = np.zeros((2, 2))

    for i in range(len(cluster)):
        for key, value in cluster[i].items():
            if key < 5:
                sign[i][0] += value 

        sign[i][1] = sum(cluster[0].values()) - sign[i][0]
    return sign

def cluster_num_elements_to_dict(kmeans, labels, verbose=False):
    """Build a dictionary showing the number of elements from each class in each cluster

    :param kmeans: the kmeans predicted labels
    :type kmeans: np.array
    :param labels: the true labels for each row in the kmeans array
    :type labels: pandas.DataFrame
    :param verbose: display the text output, defaults to False
    :type verbose: bool, optional
    :return: dictionary showing each cluster and the classes within it
    :rtype: dict
    """
    cluster = {}
    for i in range(len(np.unique(kmeans))):
        indices = ClusterIndicesNumpy(i, kmeans)
        inCluster = labels.to_numpy()[indices].flatten()

        cluster[i] = Counter(inCluster)

    if verbose:
        for i in range(len(np.unique(kmeans))):
            print("cluster-", i, "classes:- ", cluster[i])

    return cluster


def prepare_cluster_to_plot(cluster_dict, labels):
    """Convert cluster dict into list

    :param cluster_dict: the dict with number of classes in each cluster
    :type cluster_dict: dict
    :param labels: true labels for all the data
    :type labels: pandas.DataFrame
    :return: list representation of the dict
    :rtype: list
    """
    classes = []
    #2d array count of 0 class all clusters
    for i in range(len(cluster_dict.keys())):
        temp = []
        for j in range(len(np.unique(labels))):
            temp.append(cluster_dict[i].get(j, 0))
        classes.append(temp)

    classes = np.array(classes).T

    return classes
