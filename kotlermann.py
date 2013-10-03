#!/usr/bin/env python

import sys
import cPickle
import numpy as np
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
    best = np.array(best.mat.todense())[0]
    _vector_cache[word] = best
    return best

def predict_direction(space, leftword, rightword, measure):
    leftvector = word2vector(space, leftword)
    rightvector = word2vector(space, rightword)

    asym_left2right = measure(leftvector, rightvector)
    asym_right2left = measure(rightvector, leftvector)

    return int(asym_left2right > asym_right2left)

def main(args):
    space = read_space(args[0])
    judgements = read_judgements()

    gold = []
    predictions = []
    for (l, r), g in judgements:
        try:
            p = predict_direction(space, l, r, directional.__dict__[args[1]])
            gold.append(g)
            predictions.append(p)
            #print l, r, g, p
        except KeyError:
            continue

    gold = np.array(gold)
    predictions = np.array(predictions)

    errors = np.abs(gold - predictions)
    print np.sum(errors), "/", len(errors)
    inaccuracy = float(np.sum(errors)) / len(errors)
    print 1 - inaccuracy


if __name__ == '__main__':
    main(sys.argv[1:])


