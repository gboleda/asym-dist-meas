#!/usr/bin/env python

import sys
import pickle
import numpy as np
import pandas as pd
import pylab
from directional import invCL, ClarkeDE, cosine
from sklearn.metrics import average_precision_score


#v1 = v1/np.sqrt(v1.dot(v1))
#v2 = v2/np.sqrt(v2.dot(v2))

#dims = range(v1.shape[0])
#dims = range(v1.shape[0])[:20000]
#dims = set(list(v1.argsort()[-5000:]) + list(v2.argsort()[-5000:]))
#dims = v1.argsort()[-5000:]
#dims = v2.argsort()[-5000:]

#dims = set(v1.argsort()[-5000:]).difference(set(range(20000)))

#dims = set(np.random.randint(0, v1.shape[0], 5000))

#dims = list(dims)
#v1 = v1[dims]
#v2 = v2[dims]

def norm(v):
    if v.sum() == 0:
        return v
    return v / np.sqrt(v.dot(v))

def normalize_before(selector):
    def _selector(v1, v2, n):
        v1 = v1/np.sqrt(v1.dot(v1))
        v2 = v2/np.sqrt(v2.dot(v2))
        return selector(v1, v2, n)
    return _selector

def normalize_after(selector):
    def _selector(v1, v2, n):
        v1, v2 = selector(v1, v2, n)
        v1 = norm(v1)
        v2 = norm(v2)
        return v1, v2
    return _selector


def no_dim_selection(v1, v2, n):
    # ignore n
    return np.arange(len(v1))

def _highest_from_helper(v, n):
    return v.argsort()[-n:]

def first_dimensions(v1, v2, n):
    return np.arange(n)

def last_dimensions(v1, v2, n):
    return np.arange(len(v1))[-n:]

def highest_from_v1(v1, v2, n):
    return _highest_from_helper(v1, n)

def highest_from_v2(v1, v2, n):
    return _highest_from_helper(v2, n)

def highest_of_sum(v1, v2, n):
    return _highest_from_helper(v1 + v2, n)

def highest_of_mult(v1, v2, n):
    return _highest_from_helper(np.multiply(v1, v2), n)

def plot_all(results):
    pylab.clf()
    lines = results[0][1].keys()
    points_x = [np.log10(r[0]) for r in results]
    for line in lines:
        points_y = [r[1][line] for r in results]
        pylab.plot(points_x, points_y)
    pylab.legend(lines)

    pylab.show()


def add_vector_columns(data, space, selector_space=None):
    keepers = []
    word1vectors = []
    word2vectors = []
    word1selectors = []
    word2selectors = []
    if not selector_space:
        selector_space = space
    for i, row in data.iterrows():
        w1 = row['word1']
        w2 = row['word2']
        try:
            v1 = space.get_vector(w1)
            v2 = space.get_vector(w2)
            s1 = selector_space.get_vector(w1)
            s2 = selector_space.get_vector(w2)
        except KeyError:
            keepers.append(False)
            continue
        keepers.append(True)
        word1vectors.append(v1)
        word2vectors.append(v2)
        word1selectors.append(s1)
        word2selectors.append(s2)
        if i % 100 == 0:
            sys.stderr.write("Finished with %d/%d\n" % (i, len(data)))

    data = data[keepers]
    data['vector1'] = word1vectors
    data['vector2'] = word2vectors
    data['selector1'] = word1selectors
    data['selector2'] = word2selectors

    return data


def compute_maps(data, dimension_selector, similarity=invCL, n=5000):
    # compute the similarities
    sims = []
    #numdims = []
    for i, row in data.iterrows():
        v1, v2 = row['vector1'], row['vector2']
        s1, s2 = row['selector1'], row['selector2']
        dims = dimension_selector(s1, s2, n)
        sims.append(similarity(v1[dims], v2[dims]))
        #numdims.append(len(dims))
    data['sims'] = sims

    concepts = set(data['word1'])
    relations = set(data['relation'])

    aps = {}
    for r in relations:
        aps[r] = []

    for w in concepts:
        subdata = data[data['word1'] == w]
        for i, row in subdata.iterrows():
            for r in relations:
                ap = average_precision_score(subdata['relation'] == r, subdata['sims'])
                aps[r].append(ap)

    del data['sims']
    #del data['numdims']

    return {k : np.mean(a) for k, a in aps.iteritems()}

if __name__ == '__main__':
    space = pickle.load(open(sys.argv[1]))
    # make sure we're only working with items in space
    data = pd.read_table("data/smallBLESS-nopos_train.txt")
    data = add_vector_columns(data, space)

    maps = compute_maps(data, no_dim_selection)
    for r, aps in maps.iteritems():
        print "%s: %.3f" % (r, np.mean(aps))






