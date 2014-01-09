#!/usr/bin/env python
import sys
import pickle
import numpy as np
import pandas as pd
import logging
from sklearn import cross_validation, svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.utils import shuffle
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix


COLNAMES = "/var/local/roller/data/dist-spaces/CORE_SS.vectorspace.cols"
cols = np.array([l.strip().split("\t")[0] for l in open(COLNAMES)])

DATA_FILE = "data/BLESS.txt"
#pairs = [l.rstrip().replace('-j', '-a').split() for l in open(DATA_FILE)]
pairs = [l.rstrip().split() for l in open(DATA_FILE)]
space = pickle.load(open(sys.argv[1]))

# kernel options are 'linear', 'poly', 'rbf', 'sigmoid'
KERNEL = 'poly'
DEGREE = 3 # polynomial degree. ignored elsewhere
REGULARLIZATION = 1
NUM_FEATURES = 300
NUM_CROSS_VALIDATION = 20
RANDOM_SEED = 10

clf = svm.SVC(kernel=KERNEL, C=REGULARLIZATION, degree=DEGREE, tol=1.5, probability=True)
#clf = LogisticRegression(C=REGULARLIZATION, penalty='l1', tol=0.001)

clsmapping = {'random-n': 0, 'coord': 1, 'hyper': 2, 'mero': 3}
rclsm = {v : k for k, v in clsmapping.iteritems() }

def by_left_groupings(table):
    pass

X = []
Y = []
data_redone = []
byleft = {}
byright = {}
i = 0
for left, rinfo, cls_lab, right in pairs:
    if cls_lab not in clsmapping:
        continue
    cls = clsmapping[cls_lab]

    try:
        vleft = space.get_row(left).mat.A[0]
        vright = space.get_row(right).mat.A[0]
    except KeyError, e:
        logging.info("key error: %s\n" % e)
        continue

    record = (left, rinfo, cls_lab, right)
    data_redone.append(record)
    if left not in byleft:
        byleft[left] = set()
    byleft[left].add(i)
    if right not in byright:
        byright[right] = set()
    byright[right].add(i)

    if NUM_FEATURES > 0: vleft = vleft[:NUM_FEATURES]
    vleft_n = vleft / np.sqrt(vleft.dot(vleft))
    if NUM_FEATURES > 0: vright = vright[:NUM_FEATURES]
    vright_n = vright / np.sqrt(vright.dot(vright))

    diff = (vleft - vright)
    diff_n = (vleft_n - vright_n)
    #X.append(diff)
    #X.append(np.concatenate([vleft, vright, diff, vleft_n, vright_n, diff_n]))
    #X.append(np.concatenate([diff_n, diff_n ** 2]))
    X.append(np.concatenate([vleft, vright]))
    #X.append(np.concatenate([vleft_n, vright_n, diff_n, diff_n ** 2]))
    #X.append(np.concatenate([diff_n, diff_n ** 2]))
    Y.append(int(cls))
    i += 1


data_redone = np.array(data_redone)
X = np.array(X)
Y = np.array(Y)

logging.info("num data points: %d" % Y.shape)

# print "4 way classification accuracy (LOO):"
# pred_probs = []
# pred_labs = []
# for i, (train, test) in enumerate(cross_validation.KFold(len(Y), n_folds=NUM_CROSS_VALIDATION)):
#     print "   progress %.1f" % (100. * i / NUM_CROSS_VALIDATION)
#     train_X = X[train,:]
#     train_Y = Y[train]
#     test_X = X[test,:]
#     test_Y = Y[test]
#     learned = clf.fit(train_X, train_Y)
#     probs = learned.predict_proba(test_X)
#     pred_probs += list(probs)
#     pred_labs += list(learned.predict(test_X))
# print
# print "Accuracy: %.3f" % np.mean(pred_labs == Y)
# print confusion_matrix(Y, pred_labs)
# print
# results = [
#     {'left': l, 'info': i, 'relat': c, 'right': r, 'prandom': pr,
#         'pcoord': pc, 'phyper': ph, 'pmero': pm, 'pred_label': rclsm[pl]}
#     for (l, i, c, r), (pr, pc, ph, pm), pl in
#     zip(data_redone, pred_probs, pred_labs)
# ]

logging.info("4 way classification (stratified by word, %d folds)" % len(byleft))
results = []
all_pred = []
all_Y = []
all_is = set(xrange(len(Y)))
for z, (left, rinfo, cls_lab, right) in enumerate(data_redone):
    strat_left = byleft[left]
    strat_right = byright[right]

    train_idx = np.array(list(all_is.difference(strat_left.union(strat_right))))
    train_X = X[train_idx,:]
    train_Y = Y[train_idx]

    test_X = X[z,:]
    test_Y = Y[z]

    learned = clf.fit(train_X, train_Y)
    probs = learned.predict_proba(test_X)
    labs = learned.predict(test_X)

    all_pred += list(labs)
    all_Y.append(test_Y)

    data = {}
    pr, pc, ph, pm = probs[0,:]
    data['left'] = left
    data['info'] = rinfo
    data['relat'] = cls_lab
    data['right'] = right
    data['prandom'] = pr
    data['pcoord'] = pc
    data['phyper'] = ph
    data['pmero'] = pm
    data['pred_label'] = rclsm[labs[0]]
    results.append(data)

    logging.debug("  finished with %d/%d\n" % (z, len(Y)))


all_Y = np.array(all_Y)
all_pred = np.array(all_pred)
sys.stderr.write("Accuracy: %0.3f\n" % (float(np.sum(all_Y == all_pred)) / len(all_Y)))
sys.stderr.write("Accuracy: %0.3f +/0 %.3f\n" % (np.mean(scores), 2 * np.std(scores)))
results = pd.DataFrame(results)
results.to_csv(sys.stdout, index=False)

#scores = cross_validation.cross_val_score(DummyClassifier('most_frequent'), X, Y, cv=NUM_CROSS_VALIDATION)
#print "Baseline: %0.3f (+/- %.3f)" % (scores.mean(), scores.std() * 2)
#print



