def rskSplit():
    
    #from sklearn.model_selection import RepeatedStratifiedKFold
    
    classifiers = []
    scores = []
    
    for i in range(0,11):
    
        X , y = helperfn.get_data(i)

        
        rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,random_state=36851234)
            
        for train_index, test_index in rskf.split(X, y):
            X_train, X_test = np.take(X,train_index, axis=0), np.take(X,test_index, axis=0)
            y_train, y_test = np.take(y,train_index, axis=0), np.take(y,test_index, axis=0)
            
            
            classifiers = classifiers + \
                        [GaussianNB().fit(X_train.values,
                                        y_train.values)]
            scores = scores + [(classifiers[i].score( X_train, y_train),
                                        classifiers[i].score(X_test, y_test))]

       
            print("Scores for dataset: ", label_def.get(i-1, i-1))
            print("Training data score: ", scores[i][0])
            print("Testing data score: ", scores[i][1])
            print("--------------------------------------")