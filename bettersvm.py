#!/usr/bin/env python

import sys
import pickle
import numpy as np
import pandas as pd
import logging
import argparse
import datetime
from functools import partial
from random import sample
from math import ceil
from sklearn import cross_validation, svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from directional import *
from sklearn.preprocessing import normalize

logging.basicConfig(
    level=logging.DEBUG,
    format="[ %(module)-8s %(levelname)-10s %(asctime)s  %(relativeCreated)-10d ]  %(message)s",
    datefmt="%H:%M:%S:%m"
)

# kernel options are 'linear', 'poly', 'rbf', 'sigmoid'
RANDOM_SEED = 10

def vecnorm(v):
    return normalize(np.array([x]))[0]

def fixnan(x):
    if np.isnan(x):
        return 0
    else:
        return x


def eta_calculator(starttime, progress):
    delta = datetime.datetime.now() - starttime
    eta = delta.total_seconds() * ((1. - progress)/progress)
    return datetime.timedelta(seconds=eta)


def classifier_factory(classifier_type):
    if classifier_type == 'logreg':
        return LogisticRegression(penalty='l1')
    elif classifier_type == 'forest':
        return RandomForestClassifier()
    elif classifier_type == 'lsvm':
        return svm.LinearSVC()
    elif classifier_type == 'rsvm':
        return svm.SVC(cache_size=2048)
    elif classifier_type == 'svm':
        #return svm.SVC(kernel=KERNEL, degree=DEGREE, tol=1.5, probability=True)
        return svm.SVC(kernel='poly', degree=2, cache_size=2048)
        #return svm.SVC(kernel=KERNEL, degree=DEGREE, tol=1.5, cache_size=2048)
    elif classifier_type == 'dummy':
        return DummyClassifier('most_frequent')
    else:
        raise ValueError('classifier_type "%s" not implemented' % classifier_type)

def _lookup_word(space, word):
    if word in space.row2id:
        v = space.get_row(word)
    elif word.endswith('-j') and word[:-2] + '-a' in space.row2id:
        v = space.get_row(word[:-2] + '-a')
    elif word.endswith('-n') and (word[:-2].title() + '-p') in space.row2id:
        v = space.get_row(word[:-2].title() + '-p')
    else:
        for pos in ('-n', '-p', '-a', '-j'):
            if word + pos in space.row2id:
                v = space.get_row(word + pos)
                break
        else:
            return None
            raise KeyError("Couldn't find '%s' in space." % word)
    return v.mat.A[0]

def generate_features(space, word1, word2, feattype, numfeatures):
    v1 = _lookup_word(space, word1)
    v2 = _lookup_word(space, word2)

    if v1 is None or v2 is None:
        return None

    if numfeatures > 0:
        v1 = v1[:numfeatures]
        v2 = v2[:numfeatures]

    if feattype == 'cosine':
        v1, v2 = normalize([v1, v2])
        cos = v1.dot(v2)
        return np.array([cos])
    elif feattype == 'unsupervised':
        return np.array([cosine(v1, v2), lin(v1, v2), alphaSkew(v1, v2), WeedsPrec(v1, v2), ClarkeDE(v1, v2), ClarkeDE(v2, v1), invCL(v1, v2), invCL(v2, v1), projection(v1, v2), projection(v2, v1)])
    elif feattype == 'vectors':
        return np.concatenate([v1, v2])
    elif feattype == 'normvectors':
        v1, v2 = normalize([v1, v2])
        return np.concatenate([v1, v2])
    elif feattype == 'norm1vectors':
        v1, v2 = normalize([v1, v2], norm='l1')
        return np.concatenate([v1, v2])
    elif feattype.startswith('crazy.'):
        crazy, shouldnorm, method, order, window, slide = feattype.split(".")

        shouldnorm = (shouldnorm.lower() in ('yes', 'y', '1', 'norm', 'true', 't'))
        if shouldnorm:
            v1, v2 = normalize([v1, v2])

        if order == 'v1':
            z = (-v1).argsort()
            v1 = v1[z]
            v2 = v2[z]
        elif order == 'v2':
            z = (-v2).argsort()
            v1 = v1[z]
            v2 = v2[z]
        elif order == 'f':
            pass
        else:
            raise ValueError("Order '%s' not okay." % order)

        window = int(window)
        slide = int(slide)
        assert slide <= window, "Slide should not be larger than window."

        f = []

        m = np.array([v1, v2])
        d = np.min(m, axis=0)
        D = v1.shape[0]
        if method == 'diff':
            i = 0
            while i < D:
                j = min(i+window,D)
                diffwindow = v1[i:j] - v2[i:j]
                difftogether = v1[:j] - v2[:j]
                w = np.sum(diffwindow)
                t = np.sum(difftogether)
                f += [w, t, w*w, t*t]
                i += slide
        elif method == 'asym':
            i = 0
            while i < D:
                j = min(i+window,D)
                clarke1 = np.sum(d[i:j]) / np.sum(v1[i:j])
                clarke2 = np.sum(d[i:j]) / np.sum(v2[i:j])
                f += [clarke1, clarke2, np.sqrt(clarke1 * clarke2)]
                clarke3 = np.sum(d[:j]) / np.sum(v1[:j])
                clarke4 = np.sum(d[:j]) / np.sum(v2[:j])
                f += [clarke3, clarke4, np.sqrt(clarke3 * clarke4)]
                i += slide
        elif method == 'cos':
            i = 0
            while i < D:
                j = min(i+window,D)
                v1w, v2w = normalize([v1[i:j], v2[i:j]])
                v1t, v2t = normalize([v1[:j], v2[:j]])
                f += [np.dot(v1w, v2w), np.dot(v1t, v2t)]
                i += slide
        f = [fixnan(x) for x in f]
        return np.array(f)
    elif feattype == 'diffs':
        diff = v1 - v2
        return np.concatenate([diff, diff ** 2])
    elif feattype == 'norm1diffs':
        v1 = v1 / v1.sum()
        v2 = v2 / v2.sum()
        diff = v1 - v2
        return np.concatenate([diff, diff ** 2])
    elif feattype == 'normdiffscosine':
        v1 = v1 / np.sqrt(v1.dot(v1))
        v2 = v2 / np.sqrt(v2.dot(v2))
        diff = v1 - v2
        return np.concatenate([diff, np.array([cosine(v1, v2),])])
    elif feattype == 'diffhist':
        v1 = v1 / np.sqrt(v1.dot(v1))
        v2 = v2 / np.sqrt(v2.dot(v2))
        diff = v1 - v2
        from scipy import histogram
        return histogram(diff, bins=10)[0]
    elif feattype == 'random':
        return np.random.rand(1)
    elif feattype == 'normdiffs':
        v1 = v1 / np.sqrt(v1.dot(v1))
        v2 = v2 / np.sqrt(v2.dot(v2))
        diff = v1 - v2
        return np.concatenate([diff, diff ** 2])
    elif feattype == 'word1':
        return v1
    elif feattype == 'normword1':
        v1 = v1 / np.sqrt(v1.dot(v1))
        return v1
    elif feattype == 'word2':
        return v2
    elif feattype == 'normword2':
        v2 = v2 / np.sqrt(v2.dot(v2))
        return v2
    elif feattype == 'diffnorms':
        diff = v1 - v2
        mag = np.sqrt(diff.dot(diff))
        if mag == 0: mag = 1
        diff = diff/mag
        return np.concatenate([diff, diff ** 2])
    elif feattype == 'alles':
        v1n = v1 / np.sqrt(v1.dot(v1))
        v2n = v2 / np.sqrt(v2.dot(v2))
        diffn = v1n - v2n
        #return np.concatenate([v1n, np.abs(v1n), v2n, np.abs(v2n), diffn, np.abs(diffn)])
        return np.concatenate([v1n, v1n ** 2, v2n, v2n ** 2, diffn, diffn ** 2])
    else:
        #raise ValueError("Feature type '%s' not supported."  % feattype)
        return None

def add_features(dataframe, space, feature_generator, destination='features'):
    logging.info('Loading in features...')
    features = []
    for i, row in dataframe.iterrows():
        features.append(feature_generator(space, row['word1'], row['word2']))
    dataframe[destination] = features
    logging.info('Done loading features.')
    return dataframe

def add_features2(dataframe, space, feature_generator, destination='features'):
    logging.info('Loading in features...')
    features = []
    for i, row in dataframe.iterrows():
        row = dict(row)
        v1 = space.get_row(row['word1']).mat.A[0]
        v2 = space.get_row(row['word2']).mat.A[0]
        row['cosine'] = v1.dot(v2) / np.sqrt(v1.dot(v1) * v2.dot(v2))
        del row['word1']
        del row['word2']
        del row['relation']
        del row['target']
        features.append(np.array([row[v] for v in row.iterkeys()]))
    dataframe[destination] = features
    logging.info('Done loading features.')
    return dataframe


def compute_crossval_accuracy(data, klassifier, unmapper, nfolds=20):
    # shuffle things around
    s = shuffle(xrange(len(data)), random_state=RANDOM_SEED)
    # k, let's do some classifying
    scores = []
    total_right = 0
    total_total = 0
    num_steps = nfolds
    all_predictions = []
    all_answers = []
    logging.info("Starting to classify (crossval)... Splitting into %d folds." % num_steps)

    splits = []

    starttime = datetime.datetime.now()
    for i, (train, test) in enumerate(cross_validation.KFold(len(data), n_folds=nfolds, indices=True)):
        train_split = data.iloc[s[train]]
        test_split = data.iloc[s[test]]

        test_X = np.array(list(test_split['features']))
        test_Y = np.array(test_split['target'])

        train_X = np.array(list(train_split['features']))
        train_Y = np.array(train_split['target'])

        logging.debug("Learning %d/%d [%s training matrix]" % (i + 1, num_steps, train_X.shape))

        learned = klassifier.fit(train_X, train_Y)

        logging.debug("Testing %d/%d [%s testing matrix]" % (i + 1, num_steps, test_X.shape))

        percent_complete = (i + 1.) / num_steps

        #probs = learned.predict_proba(test_X)
        labels = learned.predict(test_X)
        acc = np.sum(labels == test_Y) / float(len(labels))
        total_right += np.sum(labels == test_Y)
        total_total += len(labels)
        scores.append(acc)
        all_predictions += list(labels)
        all_answers += list(test_Y)
        running_acc = np.sum(np.array(all_predictions) == np.array(all_answers)) / float(len(all_answers))

        test_split['prediction'] = labels
        test_split['prediction_l'] = map(unmapper.__getitem__, labels)
        test_split['ntraining'] = train_X.shape[0]
        test_split['nfolds'] = nfolds
        test_split['crossval'] = i + 1
        for j, k in unmapper.iteritems():
            #test_split['p_' + k] = probs[:,j]
            pass
        splits.append(test_split)
        logging.debug("Processed row %3d/%3d (%.3f); acc: %.3f; running acc: %.3f" % (i + 1, num_steps, percent_complete, acc, running_acc))
        logging.debug("ETA: %s" % eta_calculator(starttime, percent_complete))

    everything = pd.concat(splits)
    del everything['features']
    everything.to_csv(sys.stdout, index=False)

    logging.info("Done classifying!")
    logging.info("Classifier: %s" % klassifier)
    logging.info("Accuracy: %0.3f +/- %0.3f" % (np.mean(scores), 2 * np.std(scores)))
    logging.info("Accuracy: %0.3f " % (float(total_right) / total_total))
    logging.info("Confusion Matrix:")
    logging.info("         " + " ".join("%8s" % k for i, k in unmapper.iteritems()))
    confusion = confusion_matrix(all_answers, all_predictions)
    for i, k in unmapper.iteritems():
        s = ("%8s " % k) + " ".join("%8d" % v for v in confusion[i,:])
        logging.info(s)



def compute_unseen_accuracy(data, klassifier, unmapper, stratify_column='word1'):
    # shuffle things around
    #s = shuffle(xrange(len(data)), random_state=RANDOM_SEED)

    s = np.arange(len(data))
    # k, let's do some classifying
    scores = []
    total_right = 0
    total_total = 0
    num_steps = len(set(data[stratify_column]))
    all_predictions = []
    all_answers = []
    logging.info("Starting to classify... Splitting into %d folds." % num_steps)

    splits = []

    starttime = datetime.datetime.now()
    for i, (columnkey, test_split) in enumerate(data.groupby(stratify_column)):
        train_split = data[data[stratify_column] != columnkey]

        test_X = np.array(list(test_split['features']))
        test_Y = np.array(test_split['target'])

        banned_words1 = set(test_split['word1'])
        banned_words2 = set(test_split['word2'])

        word1_mask = train_split['word1'].map(lambda x: x not in banned_words1)
        word2_mask = train_split['word2'].map(lambda x: x not in banned_words2)
        both_mask = word1_mask & word2_mask
        number_banned = (len(both_mask) - np.sum(both_mask))

        train_view = train_split[both_mask]
        train_X = np.array(list(train_view['features']))
        train_Y = np.array(train_view['target'])

        logging.debug("Learning %d/%d [%s training matrix]" % (i + 1, num_steps, train_X.shape))

        learned = klassifier.fit(train_X, train_Y)

        logging.debug("Testing '%s' %d/%d [%s testing matrix]" % (columnkey, i + 1, num_steps, test_X.shape))

        percent_complete = (i + 1.) / num_steps

        #probs = learned.predict_proba(test_X)
        labels = learned.predict(test_X)
        acc = np.sum(labels == test_Y) / float(len(labels))
        total_right += np.sum(labels == test_Y)
        total_total += len(labels)
        scores.append(acc)
        all_predictions += list(labels)
        all_answers += list(test_Y)
        running_acc = np.sum(np.array(all_predictions) == np.array(all_answers)) / float(len(all_answers))

        test_split['prediction'] = labels
        test_split['prediction_l'] = map(unmapper.__getitem__, labels)
        test_split['ntraining'] = train_X.shape[0]
        test_split['nbanned'] = number_banned
        test_split['nfolds'] = num_steps
        test_split['foldno'] = i + 1
        #for j, k in unmapper.iteritems():
        #    test_split['p_' + k] = probs[:,j]
        splits.append(test_split)
        logging.debug("Processed row %3d/%3d (%0.3f); acc: %.3f; running acc: %.3f" % (i + 1, num_steps, percent_complete, acc, running_acc))
        logging.debug("ETA: %s" % eta_calculator(starttime, percent_complete))

    everything = pd.concat(splits)
    del everything['features']
    #everything.to_csv(sys.stdout, index=False)

    logging.info("Done classifying!")
    logging.info("Classifier: %s" % klassifier)
    logging.info("Accuracy: %0.3f +/- %0.3f" % (np.mean(scores), 2 * np.std(scores)))
    logging.info("Accuracy: %0.3f " % (float(total_right) / total_total))
    logging.info("Confusion Matrix:")
    logging.info("         " + " ".join("%8s" % k for i, k in unmapper.iteritems()))
    confusion = confusion_matrix(all_answers, all_predictions)
    for i, k in unmapper.iteritems():
        s = ("%8s " % k) + " ".join("%8d" % v for v in confusion[i,:])
        logging.info(s)


def justtrain(data, klassifier, space, rmapper):
    train_X = np.array(list(data['features']))
    train_Y = np.array(data['target'])
    learned = klassifier.fit(train_X, train_Y)
    coefs = learned.coef_
    inter = learned.intercept_
    for i, klass in rmapper.iteritems():
        weights = " ".join("%f" % c for c in coefs[i,:])
        print "%s\t%f %s" % (klass, inter[i], weights)



def findfeatures(data, klassifier, space, rmapper, num_components, findfeaturesmode='negpos', findfeaturesclass=None):
    train_X = np.array(list(data['features']))
    train_Y = np.array(data['target'])
    learned = klassifier.fit(train_X, train_Y)

    m = space.get_cooccurrence_matrix().get_mat()

    part1 = learned.coef_[:,:num_components]
    part2 = learned.coef_[:,num_components:]
    #s = m.argsort(axis=0)

    NUM_SELECT = 250

    lookup = {i : x for i, (x, y) in enumerate([l.strip().split("\t") for l in open(space.filename.replace(".pkl", ".cols"))])}
    tm = space.operations[1]._DimensionalityReductionOperation__transmat.get_mat().todense()
    #s = tm.argsort(axis=0)

    diff1 = tm[:,:num_components] * learned.coef_[:,:num_components].T
    diff2 = tm[:,:num_components] * learned.coef_[:,num_components:].T

    for k, klass in rmapper.iteritems():
        if findfeaturesclass and klass != findfeaturesclass:
            continue
        else:
            print "class '%s':" % klass
        g1 = diff1[:,k].A.T[0]
        g2 = diff2[:,k].A.T[0]
        if findfeaturesmode == 'neg':
            keys1 = [lookup[y] for y in g1.argsort()[:2*NUM_SELECT]]
            keys2 = []
        elif findfeaturesmode == 'negpos':
            keys1 = [lookup[y] for y in g1.argsort()[:NUM_SELECT]]
            keys2 = [lookup[y] for y in g2.argsort()[-NUM_SELECT:]]
        else:
            raise ValueError, "Can't find features mode '%s'" % findfeaturesmode

        for x in set(keys1 + keys2):
            print x
        if not findfeaturesclass:
            print



def main():
    parser = argparse.ArgumentParser(
                description='Classifies relations using a semantic space as features.')
    parser.add_argument('action', choices=('unseen', 'crossval', 'findfeatures', 'train'), default='unseen',
                        help='Action to perform.')
    parser.add_argument('-d', '--data', type=argparse.FileType('r'),
                        help='Data to classify.')
    parser.add_argument('-s', '--space', type=argparse.FileType('r'),
                        help='Vector space.')
    parser.add_argument('-t', '--target', default=-1,
                        help='Target classification field (default last field).')
    parser.add_argument('-m', '--model', help='Model type.')
    parser.add_argument('-n', '--numfeatures', type=int, default=0,
                        help='Number of vector space dimensions to use (default all).')
    parser.add_argument('-f', '--features',
                        help='Feature space for classifier.')
    parser.add_argument('-p', '--predictions', action='store_true',
                        help='Output a CSV of model predictions')
    parser.add_argument('--stratifier', default='word1', help='Column to stratify on.')
    parser.add_argument('--folds', default=20, type=int, help='Number of crossval folds')
    parser.add_argument('--findfeaturesmode', default='negpos', help='abs|pos|neg', choices=('abs', 'pos', 'neg', 'posneg', 'negpos'))
    parser.add_argument('--findfeaturesclass', help='only find features for a certain classifier')

    args = parser.parse_args()
    logging.info("Run with args '%s'" % args)

    klassifier = classifier_factory(args.model)
    feature_generator = partial(generate_features, feattype=args.features, numfeatures=args.numfeatures)

    logging.info('Reading table...')
    data = pd.read_table(args.data)
    #data = pd.read_table(args.data, names=('word1', 'word2', 'entails'))
    logging.info('Reading space...')
    space = pickle.load(args.space)
    setattr(space, 'filename', args.space.name)

    # classifier needs integers as the target field
    if isinstance(args.target, int):
        target_field = data.columns[args.target]
    else:
        target_field = args.target
    target_options = set(data[target_field])
    target_unmapper = dict(enumerate(target_options))
    target_mapper = {v : k for k, v in target_unmapper.iteritems()}
    data['target'] = map(target_mapper.__getitem__, data[target_field])

    # need to identify what are the words in the file

    # add in the features
    data = add_features2(data, space, feature_generator)
    data = data[data['features'].map(lambda x: x is not None)]
    logging.info("%d pairs with features..." % len(data))

    if args.action == 'unseen':
        compute_unseen_accuracy(data, klassifier, target_unmapper, stratify_column=args.stratifier)
    elif args.action == 'crossval':
        compute_crossval_accuracy(data, klassifier, target_unmapper, nfolds=args.folds)
    elif args.action == 'findfeatures':
        num_components = args.numfeatures and args.numfeatures or space.element_shape[0]
        findfeatures(data, klassifier, space, target_unmapper, num_components, findfeaturesmode=args.findfeaturesmode, findfeaturesclass=args.findfeaturesclass)
    elif args.action == 'train':
        justtrain(data, klassifier, space, target_unmapper)
    else:
        raise ValueError("Invalid action, '%s'!" % args.action)


if __name__ == '__main__':
    main()


