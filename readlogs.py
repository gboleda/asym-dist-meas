#!/usr/bin/env python

import sys
import pandas as pd

features_sort = {'vectors': 0, 'normvectors': 1, 'diffs': 2, 'normdiffs': 3}
model_sort = {'dummy': 0, 'svm': 1, 'logreg': 2, 'lsvm': 3}

results = []
for logfile in sys.argv[1:]:
    try:
        with open(logfile) as f:
            nicelogfile = logfile.replace("output/", "").replace(".txt.log.err", "").split("/")
            if len(nicelogfile) == 5:
                names = ("setting", "space", "features", "model", "data")
            elif len(nicelogfile) == 6:
                names = ("stratifier", "setting", "space", "features", "model", "data")
            for line in f:
                line = line.rstrip()
                if "Accuracy" in line and "+" not in line:
                    acc = line[-5:]
                    d = dict(zip(names, nicelogfile))
                    d["accuracy"] = "%0.3f" % float(acc)
                    d["sort1"] = model_sort[d["model"]]
                    d["sort2"] = features_sort[d["features"]]
                    results.append(d)
    except IOError, e:
        continue

results = pd.DataFrame(results)
results = results.sort(columns=('sort1', 'sort2'))

results.to_csv(sys.stdout, sep="\t", index=False)


