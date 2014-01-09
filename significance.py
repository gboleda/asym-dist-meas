#!/usr/bin/env python

"""
Compare the outputs of two classifiers for statistical significance.

Uses a Wilcoxon signed-rank test over the stratifier. It's important the
stratifier be the same as the the one used in training/testing.
"""

import sys
import argparse
import pandas as pd
import numpy as np
from itertools import izip, combinations
from scipy.stats import wilcoxon

def main():
    parser = argparse.ArgumentParser('Computes statistical significance between two settings.')
    parser.add_argument('csvfiles', nargs='+')
    parser.add_argument('--stratifier', '-s', default='word1', help='Stratifier used in training/testing.')
    args = parser.parse_args()

    all_scores = {}
    for csvfile in args.csvfiles:
        try:
            table = pd.read_csv(csvfile)
        except pd.parser.CParserError:
            print "had to skip %s" % csvfile
            continue
        strat = table.groupby(args.stratifier)
        accs = []
        for stratname, strat in table.groupby(args.stratifier):
            acc = np.sum(strat['target'] == strat['prediction']) / float(len(strat))
            accs.append(acc)
        all_scores[csvfile] = accs

    for leftfile, rightfile in combinations(all_scores.keys(), 2):
        print leftfile
        print rightfile
        left, right = all_scores[leftfile], all_scores[rightfile]
        t, p = wilcoxon(left, right)
        if p < .001:
            stars = "***"
        elif p < .01:
            stars = "**"
        elif p < .05:
            stars = "*"
        else:
            stars = ""
        print "la: %.3f     ra: %.3f   p: %.3f %s" % (np.mean(left), np.mean(right), p, stars)
        print




if __name__ == '__main__':
    main()

