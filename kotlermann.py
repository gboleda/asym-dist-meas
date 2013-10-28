#!/usr/bin/env python

import sys
import cPickle
import numpy as np
import scipy
import directional

#JUDGEMENTS_FILE = "kotlermann_judgements.txt"
JUDGEMENTS_FILE = "noun-noun-entailment-dataset-baroni-etal-eacl2012.txt"
#JUDGEMENTS_FILE = "noun-noun-test.txt"

# to control whether we want to do the "longest vector" trick or not (Baroni's dataset has short POS)
DATASETWITHPOS = True
if JUDGEMENTS_FILE == "kotlermann_judgements.txt":
    DATASETWITHPOS = False

def read_judgements():
    #    lines = open(JUDGEMENTS_FILE).read().split("\r\n")
    # checked that it still works with kotlermann's data:
    lines = open(JUDGEMENTS_FILE).read().split("\n")
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

    if word in _vector_cache:
        return _vector_cache[word]
    keys = space.row2id.iterkeys()
    #GB: not sure this is adequate, it's kind of ugly
    best = None

    if DATASETWITHPOS == False:
        # this is tricky, as our spaces have pos tags, and our
        # data set does not. as a heuristic now, we'll choose
        # the longest candidate vector

        candidates = [k for k in keys if k.startswith(word + "-") and len(k) == len(word) + 2]
        if not candidates:
            raise KeyError, "Didn't have %s in our db." % word

        candidate_v = [space.get_row(c) for c in candidates]
        best = max(candidate_v, key=lambda x: (x * x.transpose())[0,0])

    else:
        if word in keys:
            best = space.get_row(word)
        else:
            raise KeyError, "Didn't have %s in our db." % word

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
        print "%-30s %-15s %5.3f %5.3f %5.3f" % (
                out_filename, measure.func_name, average_precision, p_at_1000, r_at_1000)
    print

def evaluate_space_accuracy_balAPinc(space_filename, threshold):
    # don't forget to clear our vector cache
    global _vector_cache
    _vector_cache = {}
    space = read_space(space_filename)
    judgements = read_judgements()
    
    oov = 0
    measure = MEASURES[6]
    if measure.func_name != 'balAPinc':
        sys.exit("oops! Not evaluating balAPinc, but %s", measure.func_name)
    gold = []
    predictions = []
    out_filename = space_filename[-30:]

    for (l, r), g in judgements:
        try:
            score = asym_measure(space, l, r, measure)
            predictions.append(score)
            gold.append(g)
        except KeyError:
            # ignore OOV situations
            oov += 1
            continue

    gold = np.array(gold, dtype = bool) # convert to boolean
    predictions = np.array(predictions)
    predbool = predictions > threshold # this is also boolean
    matches = float( ( (gold == False) & (predbool == False) ).sum() +  ( (gold == True) & (predbool == True) ).sum() )
    accuracy = matches / gold.size

    # print "gold", gold
    # print "predictions", predictions
    # print "predbool", predbool
    # print "matches", matches
    
    print "%-30s %-15s %5.2f    %d" % (
        out_filename, measure.func_name, accuracy, oov)
    print

    # outputs evaluation measures for all semantic spaces given as arguments
if __name__ == '__main__':
    if JUDGEMENTS_FILE == "kotlermann_judgements.txt":
        print "space                            measure            AP      P@1000   R@1000"
        for space_filename in sys.argv[1:]:
            evaluate_space(space_filename)
    # elif JUDGEMENTS_FILE == "noun-noun-entailment-dataset-baroni-etal-eacl2012.txt":
    else:
        threshold = 0.26 # hard-coded, taken from baroni et al eacl 2012
        print "space                            measure            Accuracy      OOV"
        for space_filename in sys.argv[1:]:
            evaluate_space_accuracy_balAPinc(space_filename,threshold)
