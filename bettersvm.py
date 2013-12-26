#!/usr/bin/env python

import sys
import pickle
import numpy as np
import pandas as pd
import logging
import argparse
from functools import partial
from sklearn import cross_validation, svm
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

logging.basicConfig(
    level=logging.DEBUG,
    format="[ %(module)-8s %(levelname)-10s %(asctime)s  %(relativeCreated)-10d ]  %(message)s",
    datefmt="%H:%M:%S:%m"
)

# kernel options are 'linear', 'poly', 'rbf', 'sigmoid'
KERNEL = 'poly'
DEGREE = 3 # polynomial degree. ignored elsewhere
REGULARLIZATION = 1
NUM_CROSS_VALIDATION = 50
RANDOM_SEED = 10


def classifier_factory(classifier_type):
    if classifier_type == 'logreg':
        return LogisticRegression(C=REGULARLIZATION, penalty='l1', tol=0.001)
    elif classifier_type == 'svm':
        #return svm.SVC(kernel=KERNEL, C=REGULARLIZATION, degree=DEGREE, tol=1.5, probability=True)
        return svm.SVC(kernel=KERNEL, C=REGULARLIZATION, degree=DEGREE, tol=1.5)
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

    if feattype == 'vectors':
        return np.concatenate([v1, v2])
    elif feattype == 'normvectors':
        v1 = v1 / np.sqrt(v1.dot(v1))
        v2 = v2 / np.sqrt(v2.dot(v2))
        return np.concatenate([v1, v2])
    elif feattype == 'diffs':
        diff = v1 - v2
        return np.concatenate([diff, diff ** 2])
    elif feattype == 'normdiffs':
        v1 = v1 / np.sqrt(v1.dot(v1))
        v2 = v2 / np.sqrt(v2.dot(v2))
        diff = v1 - v2
        return np.concatenate([diff, diff ** 2])
    else:
        #raise ValueError("Feature type '%s' not supported."  % feattype)
        return None

def add_features(dataframe, space, feature_generator):
    logging.info('Loading in features...')
    features = []
    for i, row in dataframe.iterrows():
        features.append(feature_generator(space, row['word1'], row['word2']))
    dataframe['features'] = features
    logging.info('Done loading features.')
    return dataframe

def main():
    parser = argparse.ArgumentParser(
                description='Classifies relations using a semantic space as features.')
    parser.add_argument('-d', '--data', type=argparse.FileType('r'),
                        help='Data to classify.')
    parser.add_argument('-s', '--space', type=argparse.FileType('r'),
                        help='Vector space.')
    parser.add_argument('-t', '--target', default=-1,
                        help='Target classification field (default last field).')
    parser.add_argument('-m', '--model', choices=('svm', 'logreg', 'dummy'),
                        help='Model type.')
    parser.add_argument('-n', '--numfeatures', type=int, default=0,
                        help='Number of vector space dimensions to use (default all).')
    parser.add_argument('-f', '--features', choices=('vectors', 'normvectors', 'diffs', 'normdiffs'),
                        help='Feature space for classifier.')

    args = parser.parse_args()

    klassifier = classifier_factory(args.model)
    feature_generator = partial(generate_features, feattype=args.features, numfeatures=args.numfeatures)

    logging.info('Reading table...')
    data = pd.read_table(args.data, names=('word1', 'info', 'relation', 'word2'))
    logging.info('Reading space...')
    space = pickle.load(args.space)

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
    data = add_features(data, space, feature_generator)
    data = data[data['features'].map(lambda x: x is not None)]
    logging.info("%d pairs to classify..." % len(data))

    # shuffle things around
    #s = shuffle(xrange(len(data)), random_state=RANDOM_SEED)
    s = np.arange(len(data))

    # k, let's do some classifying
    logging.info("Starting to classify... Splitting into %d folds." % NUM_CROSS_VALIDATION)
    scores = []
    total_right = 0
    total_total = 0
    for i, (train, test) in enumerate(cross_validation.KFold(len(data), n_folds=NUM_CROSS_VALIDATION, indices=True)):
        train_split = data.iloc[s[train]]
        test_split = data.iloc[s[test]]

        test_X = np.array(list(test_split['features']))
        test_Y = np.array(test_split['target'])

        banned_words1 = set(test_split['word1'])
        banned_words2 = set(test_split['word2'])

        word1_mask = train_split['word1'].map(lambda x: x not in banned_words1)
        word2_mask = train_split['word2'].map(lambda x: x not in banned_words2)
        #print np.sum(word1_mask), np.sum(word2_mask)
        mask = word1_mask & word2_mask
        train_view = train_split[mask]
        train_X = np.array(list(train_view['features']))
        train_Y = np.array(train_view['target'])

        logging.debug("Learning %d/%d [%s training matrix]" % (i + 1, NUM_CROSS_VALIDATION, train_X.shape))

        learned = klassifier.fit(train_X, train_Y)

        logging.debug("Testing  %d/%d [%s testing matrix]" % (i + 1, NUM_CROSS_VALIDATION, test_X.shape))

        #probs = learned.predict_proba(test_X)
        labels = learned.predict(test_X)
        acc = np.sum(labels == test_Y) / float(len(labels))
        total_right += np.sum(labels == test_Y)
        total_total += len(labels)
        scores.append(acc)

        logging.debug("Processed row %3d/%3d (%2.1f); acc: %.3f; running acc: %.3f" % (i + 1, NUM_CROSS_VALIDATION, 100. * (i + 1.) / NUM_CROSS_VALIDATION, acc, float(total_right)/total_total))

    logging.info("Done classifying!")
    logging.info("Classifier: %s" % klassifier)
    logging.info("Accuracy: %0.3f +/- %0.3f" % (np.mean(scores), 2 * np.std(scores)))
    logging.info("Accuracy: %0.3f +/- %0.3f" % (np.mean(scores), 2 * np.std(scores)))


if __name__ == '__main__':
    main()


