import numpy as np
from sklearn import datasets, tree, metrics, ensemble, svm

digits = datasets.load_digits()

flag_3_8 = (digits.target == 3) + (digits.target == 8)

images = digits.images[flag_3_8]
labels = digits.target[flag_3_8]

images = images.reshape(images.shape[0], -1)

n_samples = len(flag_3_8[flag_3_8])
train_size = int(n_samples * 3 / 5)
# classifier = tree.DecisionTreeClassifier()
# classifier = ensemble.RandomForestClassifier(n_estimators=20, max_depth=3, criterion="gini")
# estimator = tree.DecisionTreeClassifier(max_depth=3)
# classifier = ensemble.AdaBoostClassifier(base_estimator=estimator, n_estimators=20)
classifier = svm.SVC
(C=1.0, gamma=0.001)
classifier.fit(images[:train_size], labels[:train_size])

expected = labels[train_size:]
predicted = classifier.predict(images[train_size:])

print('Accuracy:\n', metrics.accuracy_score(expected, predicted))
print('\nConfusion matrix:\n', metrics.confusion_matrix(expected, predicted))
print('\nPrecision:\n', metrics.precision_score(expected, predicted, pos_label=3))
print('\nRecall:\n', metrics.recall_score(expected, predicted, pos_label=3))
print('\nF-measure:\n', metrics.f1_score(expected, predicted, pos_label=3))
