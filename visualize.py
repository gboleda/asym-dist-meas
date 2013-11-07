#!/usr/bin/env python

import sys
import cPickle
import numpy as np
import pylab
from matplotlib.backends.backend_pdf import PdfPages
from itertools import izip, count

COLORS = [["#666666", "#66cc66"], ["#666666", "#cc6666"]]

CIRCLECHART = True
OVERLAYED_BARCHART = False
STACKED_BARCHART = True
SCATTERPLOT = True
DIFFERENCE_HIST = False
SAMESIGN_HIST = False
PLOT_ONLY_SELECT_FEATURES = False

NUM_PAIRS = 10
NUM_TOP_FEATS = 10
NUM_DISCRIM_FEATS = 20


DATA_FILE = "data/randomized-entailing-baronietal.txt"
#DATA_FILE = "data/randomized-nonentailing-baronietal.txt"

COLORS = COLORS["nonentail" in DATA_FILE]

pairs = [l.rstrip().split() for l in open(DATA_FILE)][:NUM_PAIRS]

space = cPickle.load(open(sys.argv[1]))

def toarray(vm):
    return vm.A[0]
    try:
        return vm.toarray()[0]
    except NameError:
        return vm.A[0]


pylab.axhline(y=0, color="black")
pylab.axvline(x=0, color="black")

pp = PdfPages('multipage.pdf')

all_left = []
all_right = []
all_differences = []
all_samesigns = []
for left, right in pairs:
    vleft = toarray(space.get_row(left).mat)
    vright = toarray(space.get_row(right).mat)
    vlen = vleft.shape[0]

    # select only what I think might be discriminating features
    features = set()
    # largest magnitude features of vleft
    features.update(np.absolute(vleft).argsort()[-NUM_TOP_FEATS:])
    # largest magnitude features of vright
    features.update(np.absolute(vright).argsort()[-NUM_TOP_FEATS:])
    # features with largest difference in the vectors
    features.update(np.absolute(vleft - vright).argsort()[-NUM_DISCRIM_FEATS:])
    features = list(features)

    dimnames = space.get_id2column()
    if dimnames:
        names = [dimnames[i] for i in features]
    else:
        names = ["dim_%03d" % i for i in features]

    vleft_s = vleft[features]
    vright_s = vright[features]

    if PLOT_ONLY_SELECT_FEATURES:
        vleft = vleft_s
        vright = vright_s


    all_left += list(vleft)
    all_right += list(vright)

    factor = max(max(vleft), max(vright)) / 150.
    if CIRCLECHART:
        colors = [COLORS[u >= 0] for u in vleft_s]
        pylab.scatter(np.arange(len(features)) + 1, [0.35] * len(features), s = np.absolute(vleft_s) / factor, color=COLORS[0])
        colors = [COLORS[v >= 0] for v in vright_s]
        pylab.scatter(np.arange(len(features)) + 1, [0.45] * len(features), s = np.absolute(vright_s) / factor, color=COLORS[1])
        pylab.ylim(0, 1)
        pylab.xlim(0, len(features) + 1)
        for i, n in enumerate(names):
            pylab.text(i + 1, 0.5, n, rotation=45, fontsize=10, ha='left', va='bottom')
        pylab.title("Circle comparison " + left + " (bottom) & " + right + " (top)")
        pylab.savefig(pp, format='pdf')
        pylab.clf()


    if OVERLAYED_BARCHART:
        # overlayed bar charts
        mask = [abs(l) > abs(r) for l, r in izip(vleft_s, vright_s)]
        maxes_left = [(i, l) for i, m, l in zip(count(), mask, vleft_s) if m]
        maxes_right = [(i, r) for i, m, r in zip(count(), mask, vright_s) if not m]
        mins_left = [(i, l) for i, m, l in zip(count(), mask, vleft_s) if not m]
        mins_right = [(i, r) for i, m, r in zip(count(), mask, vright_s) if m]

        pylab.axhline(color="black")
        pylab.xlim(0, len(features))
        pylab.bar(*zip(*maxes_left), facecolor=COLORS[0], linewidth=0)
        pylab.bar(*zip(*maxes_right), facecolor=COLORS[1], linewidth=0)
        pylab.bar(*zip(*mins_left), facecolor=COLORS[0], linewidth=0)
        pylab.bar(*zip(*mins_right), facecolor=COLORS[1], linewidth=0)

        for i, n in enumerate(names):
            pylab.text(i + 0.5, 0, n, rotation=90, fontsize=10, ha='center', va='bottom')

        pylab.title("Overlayed " + left + " & " + right)
        pylab.savefig(pp, format='pdf')
        pylab.clf()

    if STACKED_BARCHART:
        # stacked bar charts
        stack_start = [((u >= 0 and v >= 0) or (u <= 0 and v <= 0)) and u or 0 for u, v in zip(vleft_s, vright_s)]
        pylab.axhline(color="black")
        pylab.xlim(0, len(features))
        pylab.bar(range(len(features)), vleft_s, facecolor=COLORS[0], linewidth=0)
        pylab.bar(range(len(features)), vright_s, bottom=stack_start, facecolor=COLORS[1], linewidth=0)
        for i, n in enumerate(names):
            pylab.text(i + 0.5, 0, n, rotation=90, fontsize=10, ha='center', va='bottom')

        pylab.title("Stacked " + left + " & " + right)
        pylab.savefig(pp, format='pdf')
        pylab.clf()


    if SCATTERPLOT:
        # scatter plot
        pylab.axhline(color="black")
        pylab.axvline(color="black")
        pylab.scatter(vleft, vright, color=COLORS[1], s=4)
        pylab.title(left + " & " + right)
        pylab.savefig(pp, format='pdf')
        pylab.clf()

    if DIFFERENCE_HIST:
        # histogram of differences
        pylab.hist(vleft - vright, bins=10, color=COLORS[1])
        all_differences += (vleft - vright)
        pylab.title(left + " & " + right + " (difference histogram)")
        pylab.savefig(pp, format='pdf')
        pylab.clf()

    if SAMESIGN_HIST:
        # histogram of same sign
        same_signs = [((u >= 0 and v >= 0) or (u <= 0 and v <= 0)) for u, v in zip(vleft_s, vright_s)]
        all_samesigns += same_signs
        pylab.hist(same_signs, bins=2, color=COLORS[1])
        pylab.title(left + " & " + right + " (signs histogram)")
        pylab.savefig(pp, format='pdf')
        pylab.clf()

if SCATTERPLOT:
    pylab.axhline(color="black")
    pylab.axvline(color="black")
    pylab.scatter(all_left, all_right, color=COLORS[1], s=1)
    pylab.title('All pairs (%s)' % DATA_FILE)
    pylab.savefig(pp, format='pdf')
    pylab.clf()

if DIFFERENCE_HIST:
    pylab.hist(all_differences, bins=30, color=COLORS[1])
    pylab.title("All Pairs [%s] (difference histogram)" % DATA_FILE)
    pylab.savefig(pp, format='pdf')
    pylab.clf()

if SAMESIGN_HIST:
    pylab.hist(all_samesigns, bins=2, color=COLORS[1])
    pylab.title("All Pairs [%s] (signs histogram)" % DATA_FILE)
    pylab.savefig(pp, format='pdf')
    pylab.clf()

pp.close()

