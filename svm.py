#!/usr/bin/env python
import sys
import pickle
import numpy as np
from sklearn import cross_validation, svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.utils import shuffle

COLNAMES = "/var/local/roller/data/dist-spaces/CORE_SS.vectorspace.cols"
cols = np.array([l.strip().split("\t")[0] for l in open(COLNAMES)])

DATA_FILE = "data/noun-noun-entailment-dataset-baroni-etal-eacl2012.txt"
pairs = [l.rstrip().split() for l in open(DATA_FILE)]
space = pickle.load(open(sys.argv[1]))

# kernel options are 'linear', 'poly', 'rbf', 'sigmoid'
KERNEL = 'poly'
DEGREE = 3 # polynomial degree. ignored elsewhere
REGULARLIZATION = 1
NUM_FEATURES = 300
NUM_CROSS_VALIDATION = 10
RANDOM_SEED = 10

#clf = svm.SVC(kernel=KERNEL, C=REGULARLIZATION, degree=DEGREE, tol=1.5)
clf = LogisticRegression(C=REGULARLIZATION, penalty='l1', tol=0.001)

X = []
Y = []
for left, right, cls in pairs:
    vleft = space.get_row(left).mat.A[0]
    if NUM_FEATURES > 0: vleft = vleft[:NUM_FEATURES]
    vleft_n = vleft / np.sqrt(vleft.dot(vleft))
    vright = space.get_row(right).mat.A[0]
    if NUM_FEATURES > 0: vright = vright[:NUM_FEATURES]
    vright_n = vright / np.sqrt(vright.dot(vright))

    diff = (vleft - vright)
    diff_n = (vleft_n - vright_n)
    #X.append(diff)
    #X.append(np.concatenate([vleft, vright, diff, vleft_n, vright_n, diff_n]))
    X.append(np.concatenate([diff_n, diff_n ** 2]))
    #X.append(np.concatenate([vleft, vright]))
    #X.append(np.concatenate([vleft_n, vright_n, diff_n, diff_n ** 2]))
    #X.append(np.concatenate([diff_n, diff_n ** 2]))
    Y.append(int(cls))

X = np.array(X)
Y = np.array(Y)

X, Y = shuffle(X, Y, random_state=RANDOM_SEED)

print "data loaded"

#print X

print "running classifier..."
I = 200
scores = cross_validation.cross_val_score(clf, X, Y, cv=NUM_CROSS_VALIDATION)
model = clf.fit(X[:-I,:], Y[:-I])

V = space.operations[1]._DimensionalityReductionOperation__transmat.to_dense_matrix().mat
V = V[:,:NUM_FEATURES]

weights1, weights2 = model.coef_[:,:NUM_FEATURES], model.coef_[:,NUM_FEATURES:]
words_weights1 = V.dot(weights1.T).A[:,0]
s = words_weights1.argsort()
for i in list(s[:10]) + list(s[-10:]):
    #print "%-5.3f    %s" % (words_weights1[i], cols[i])
    pass

words_weights2 = V.dot(weights2.T).A[:,0]
s = words_weights2.argsort()
for i in list(s[:10]) + list(s[-10:]):
    #print "%-5.3f    %s" % (words_weights2[i], cols[i])
    pass


#probs = model.predict_proba(X[-I:,:])[:,1]
#p, r, t = precision_recall_curve(Y[-I:], probs)

#import pylab
#pylab.scatter(t, r[:-1])
#pylab.show()

#print model.coef_

print "results..."
print scores
print
print "Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)





