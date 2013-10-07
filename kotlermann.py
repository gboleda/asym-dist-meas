#!/usr/bin/env python

import sys
import cPickle
import numpy as np
import scipy
import directional

JUDGEMENTS_FILE = "kotlermann_judgements.txt"
def read_judgements():
    lines = open(JUDGEMENTS_FILE).read().split("\r\n")
    cases = [l.strip().split("\t") for l in lines if l.strip()]
    judgements = [(c[:-1], int(c[-1])) for c in cases]
    # skip judgements involving more than one word
    judgements = [
        ((l, r), d) for ((l, r), d) in judgements
        if " " not in l and " " not in r
    ]
    return judgements

def read_space(filename):
    return cPickle.load(open(filename))

# let's cache our lookups because each lookup is kinda slow,
# but we look up the same word many times
_vector_cache = {}
def word2vector(space, word):
    # this is tricky, as our spaces have pos tags, and our
    # data set does not. as a heuristic now, we'll choose
    # the longest candidate vector
    if word in _vector_cache:
        return _vector_cache[word]

    keys = space.row2id.iterkeys()
    candidates = [k for k in keys if k.startswith(word + "-") and len(k) == len(word) + 2]
    if not candidates:
        raise KeyError, "Didn't have %s in our db." % word

    candidate_v = [space.get_row(c) for c in candidates]
    best = max(candidate_v, key=lambda x: (x * x.transpose())[0,0])
    if isinstance(best.mat, np.matrixlib.defmatrix.matrix):
        best = np.array(best.mat)[0]
    elif isinstance(best.mat, scipy.sparse.csr.csr_matrix):
        best = np.array(best.mat.todense())[0]
    else:
        raise ValueError, "Cannot turn a %s into a numpy array." % type(best.mat)
    _vector_cache[word] = best
    return best

def asym_measure(space, leftword, rightword, measure):
    leftvector = word2vector(space, leftword)
    rightvector = word2vector(space, rightword)
    return measure(leftvector, rightvector)

MEASURES = [
    directional.lin,
    directional.alphaSkew,
    directional.WeedsPrec,
    directional.balPrec,
    directional.ClarkeDE,
    directional.APinc,
    directional.balAPinc
]


def evaluate_space(space_filename):
    # don't forget to clear our vector cache
    global _vector_cache
    _vector_cache = {}
    space = read_space(space_filename)
    judgements = read_judgements()


    for m_i, measure in enumerate(MEASURES):
        gold = []
        predictions = []

        for (l, r), g in judgements:
            try:
                score = asym_measure(space, l, r, measure)
                predictions.append(score)
                gold.append(g)
            except KeyError:
                # ignore OOV situations
                continue

        gold = np.array(gold)
        predictions = np.array(predictions)

        average_precision = directional.AP(predictions, gold)

        cutoff_1000_score = np.sort(predictions)[-1000]

        relevant = (predictions >= cutoff_1000_score)
        p_at_1000 = np.sum(gold[relevant]) / float(np.sum(relevant))
        r_at_1000 = np.sum(gold[relevant]) / float(np.sum(gold))

        if m_i == 0:
            out_filename = space_filename[-30:]
        else:
            out_filename = ""
        print "%-30s   %-15s %5.3f       %5.3f    %5.3f" % (
                out_filename, measure.func_name, average_precision, p_at_1000, r_at_1000)
    print



if __name__ == '__main__':
    print "space                            measure            AP      P@1000   R@1000"
    for space_filename in sys.argv[1:]:
        evaluate_space(space_filename)


