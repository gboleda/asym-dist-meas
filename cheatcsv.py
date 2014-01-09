#!/usr/bin/env python

import sys
import argparse
import pandas as pd
import numpy as np

def nice_model(model):
    if model == 'svm':
        return 'SVM'
    elif model == 'logreg':
        return 'LogReg'
    elif model == 'lsvm':
        return 'LinearSVM'
    else:
        raise ValueError, "wtf is %s?" % model

def nice_features(features):
    if features == 'normdiffs': return 'Norm Diffs'
    elif features == 'diffs': return 'Vector Diffs'
    elif features == 'vectors': return 'Vectors'
    elif features == 'normvectors': return 'Norm Vectors'
    else: raise ValueError, 'wtf is %s?' % features

def main():
    parser = argparse.ArgumentParser("Outputs a CSV file for input to the prexisting knowledge grapher.")
    parser.add_argument("--stratifier", "-s", default="word1", help="The stratifier used for these channels.")
    parser.add_argument("CSVs", nargs='+', help='Output CSV files from the runs.')
    args = parser.parse_args()

    output = []
    for csv in args.CSVs:

        nicename = csv.replace("output/", "").replace("cheating/", "").replace("unseen/", "").replace(args.stratifier + "/", "")
        space, features, model, basename = nicename.split("/")

        actual_cheat_factor = []
        goal_cheat_factor = []
        accuracies = []
        try:
            table = pd.read_csv(csv)
        except pd.parser.CParserError:
            continue
        for stratname, stratgroup in table.groupby(args.stratifier):
            acc = np.sum(stratgroup['target'] == stratgroup['prediction']) / float(len(stratgroup))
            accuracies.append(acc)
            ntrain = stratgroup['ntraining'].iloc[0]
            ncheat = stratgroup['ncheats'].iloc[0]
            cheat_percent = float(ncheat) / ntrain
            actual_cheat_factor.append(cheat_percent)
            cheat_goal = stratgroup['percent_cheats_requested'].iloc[0]
            goal_cheat_factor.append(cheat_goal)

        avgacc = np.mean(accuracies)
        #stdacc = np.std(accuracies)
        stdacc = np.sqrt(avgacc * (1 - avgacc) / len(accuracies))
        output.append(dict(
            logname=csv,
            model=model,
            stratifier=args.stratifier,
            space=space,
            basename=basename,
            features=features,
            setting_name=nice_model(model) + " - " + nice_features(features),
            accuracy_mean=avgacc,
            accuracy_low=max(avgacc - 2 * stdacc, 0),
            accuracy_high=min(1.0, avgacc + 2 * stdacc),
            accuracy_std=stdacc,
            n=len(accuracies),
            cheat_goal=np.mean(goal_cheat_factor),
            cheat_mean=np.mean(actual_cheat_factor)
        ))

    pd.DataFrame(output).to_csv(sys.stdout, index=False)



if __name__ == '__main__':
    main()
