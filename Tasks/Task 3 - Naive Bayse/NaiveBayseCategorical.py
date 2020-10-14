from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
import helperfn


def build_nbc_models(test_size=0.2, random_state=0):
    """Build and score naive bayse categorical model

    :param test_size: the percentage of the sample size to test with, defaults to 0.2
    :type test_size: float, optional
    :param random_state: the random seed, defaults to 0
    :type random_state: int, optional
    :return: Tuple of all scores and classifer
    :rtype: Tuple
    """
    training_smpl = helperfn.get_data_noresults()
    raw_data_results = []

    train_test_data = []
    classifiers = []
    scores = []

    for i in range(-1, 10):
        raw_data_results = raw_data_results + [helperfn.get_results(i)]
        train_test_data = train_test_data + [train_test_split(training_smpl, raw_data_results[i], test_size=test_size, random_state=random_state)]
        classifiers = classifiers + [CategoricalNB().fit(train_test_data[i][0], train_test_data[i][2])]
        scores = scores + [(classifiers[i].score(train_test_data[i][0], train_test_data[i][2]),
                            classifiers[i].score(train_test_data[i][1], train_test_data[i][3]))]

    for i in range(len(scores)):
        print("Scores for dataset: ", i-1)
        print("Training data score: ", scores[i][0])
        print("Testing data score: ", scores[i][1])
        print("--------------------------------------")

    return classifiers, scores


