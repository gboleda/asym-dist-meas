#!/usr/bin/env python

import sys
import pickle
import numpy as np
import pandas as pd
from directional import invCL, ClarkeDE
from sklearn.metrics import average_precision_score

data = pd.read_table("data/smallBLESS-nopos.txt")
space = pickle.load(open(sys.argv[1]))
space_raw = pickle.load(open(sys.argv[2]))

sims = []
keepers = []
num_dims = []
for i, row in data.iterrows():
    w1 = row['word1']
    w2 = row['word2']
    try:
        v1 = space.get_row(w1).get_mat().A[0]
        v2 = space.get_row(w2).get_mat().A[0]


        r1 = space_raw.get_row(w1)
        r2 = space_raw.get_row(w2)

        v12 = space.operations[0].project(r1 + r2).get_mat().A[0]

    except KeyError:
        keepers.append(False)
        continue

    v1 = v1/np.sqrt(v1.dot(v1))
    v2 = v2/np.sqrt(v2.dot(v2))

    #dims = range(v1.shape[0])
    #dims = range(v1.shape[0])[:20000]
    #dims = set(list(v1.argsort()[-5000:]) + list(v2.argsort()[-5000:]))
    dims = set(list(v1.argsort()[-5000:]) + list(v12.argsort()[-5000:]))
    #dims = v1.argsort()[-5000:]
    #dims = v2.argsort()[-5000:]
    #dims = v12.argsort()[-5000:]

    #dims = set(v1.argsort()[-5000:]).difference(set(range(20000)))
    #dims = set(v12.argsort()[-5000:]).difference(set(v2.argsort()[-5000:]))

    #dims = set(np.random.randint(0, v1.shape[0], 5000))

    dims = list(dims)
    v1 = v1[dims]
    v2 = v2[dims]
    keepers.append(True)
    rel = row['relation']

    num_dims.append(len(v1))

    inv = invCL(v1, v2)
    sims.append(inv)

data = data[keepers]
data['invCL'] = sims

concepts = set(data['word1'])
relations = set(data['relation'])

maps = {}
for r in relations:
    maps[r] = []

for w in concepts:
    subdata = data[data['word1'] == w]
    for i, row in subdata.iterrows():
        for r in relations:
            ap = average_precision_score(subdata['relation'] == r, subdata['invCL'])
            maps[r].append(ap)

print "mean num dims: %f" % np.mean(num_dims)
for r, aps in maps.iteritems():
    print "%s: %.3f" % (r, np.mean(aps))






