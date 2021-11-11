import os

import numpy as np

from sklearn.feature_selection import r_regression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn import clone
from sklearn.metrics import accuracy_score

# preparing classifiers

clfs = dict()

metrics = ['minkowski', 'euclidean']
k_values = [3, 5, 7]

for k in k_values:
    for metric in metrics:
        clfs[f'K{k}_{metric}'] = KNeighborsClassifier(n_neighbors=k, metric=metric)


dataset = np.genfromtxt("wisconsin.csv", delimiter=",")
X = dataset[1:, :-1]
y = dataset[1:, -1].astype(int)
ranking = r_regression(X, y, center=True)

for i in range(1, len(ranking)+1):

    print(f'='*26 +
          f'\n{i} feature(s):')

    # getting ids of i columns with highest rankings
    ind = np.argpartition(ranking, -i)[-i:]
    # adding
    ind = np.append(ind, len(ranking))

    # new dataset with fewer features
    new_dataset = dataset[:, ind]
    X = new_dataset[1:, :-1]
    y = new_dataset[1:, -1].astype(int)

    # cross validation if more than one feature
    if i > 1:
        if i < 5:
            # if there are less than 5 features, n_splits are equal to i
            n_splits = i
        else:
            n_splits = 5
        n_repeats = 2
        rskf = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=123)

        scores = np.zeros((len(clfs), n_splits * n_repeats))

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            for clf_id, clf_name in enumerate(clfs):
                clf = clone(clfs[clf_name])
                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])
                scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)
    else:
        # if one feature then we just split dataset
        scores = np.zeros((len(clfs), 1))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=.30,
            random_state=123
        )
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            scores[clf_id, 0] = accuracy_score(y_test, y_pred)

    mean = np.mean(scores, axis=1)  # arithmetic mean of accuracies
    std = np.std(scores, axis=1)    # standard deviation

    for clf_id, clf_name in enumerate(clfs):
        print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))

    if not os.path.isdir('results'):
        os.mkdir('results')
    np.save(os.path.join('results', f'results_{i}'), scores)

