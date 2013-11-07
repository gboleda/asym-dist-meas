#!/usr/bin/env python
import sys
import cPickle
import numpy as np
from sklearn import cross_validation, svm

DATA_FILE = "data/noun-noun-entailment-dataset-baroni-etal-eacl2012.txt"
pairs = [l.rstrip().split() for l in open(DATA_FILE)]
space = cPickle.load(open(sys.argv[1]))

# kernel options are 'linear', 'poly', 'rbf', 'sigmoid'
KERNEL = 'poly'
DEGREE = 1 # polynomial degree. ignored elsewhere
REGULARLIZATION = 1
NUM_FEATURES = 300
NUM_CROSS_VALIDATION = 20

X = []
Y = []
for left, right, cls in pairs:
    vleft = space.get_row(left).mat.A[0]
    vright = space.get_row(right).mat.A[0]

    diff = (vleft - vright)[:NUM_FEATURES]
    X.append(diff)
    Y.append(int(cls))

print "data loaded"
X = np.array(X)
Y = np.array(Y)

print "running classifier..."
clf = svm.SVC(kernel=KERNEL, C=REGULARLIZATION, degree=DEGREE)
scores = cross_validation.cross_val_score(clf, X, Y, cv=NUM_CROSS_VALIDATION)

print "results..."
print scores
print
print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)





