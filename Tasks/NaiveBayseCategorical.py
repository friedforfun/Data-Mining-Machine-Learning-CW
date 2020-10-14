from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
import helperfn


training_smpl = helperfn.get_data_noresults()
raw_data_results = []

train_test_results = []
classifiers = []
scores = []
for i in range(-1, 10):
    raw_data_sets = raw_data_sets + [helperfn.get_data(i)]
    train_test_data = train_test_data + \
        [train_test_split(raw_data_sets[i][0], raw_data_sets[i]
                          [1], test_size=0.2, random_state=0)]
    classifiers = classifiers + \
        [CategoricalNB().fit(train_test_data[i][0], train_test_data[i][2])]
    scores = scores + [(classifiers[i].score(train_test_data[i][0], train_test_data[i][2]),
                        classifiers[i].score(train_test_data[i][1], train_test_data[i][3]))]

for i in range(len(scores)):
    print("Scores for dataset: ", i-1)
    print("Training data score: ", scores[i][0])
    print("Testing data score: ", scores[i][1])
    print("--------------------------------------")


