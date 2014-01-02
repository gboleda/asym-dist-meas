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
NUM_CROSS_VALIDATION = 50
RANDOM_SEED = 10


def classifier_factory(classifier_type):
    if classifier_type == 'logreg':
        return LogisticRegression(penalty='l1')
    elif classifier_type == 'svm':
        #return svm.SVC(kernel=KERNEL, degree=DEGREE, tol=1.5, probability=True)
        return svm.SVC(kernel=KERNEL, degree=DEGREE, tol=1.5)
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

def compute_unseen_accuracy(data, klassifier):
    # shuffle things around
    #s = shuffle(xrange(len(data)), random_state=RANDOM_SEED)

    s = np.arange(len(data))
    # k, let's do some classifying
    logging.info("Starting to classify... Splitting into %d folds." % NUM_CROSS_VALIDATION)
    scores = []
    total_right = 0
    total_total = 0
    #for i, (train, test) in enumerate(cross_validation.KFold(len(data), n_folds=NUM_CROSS_VALIDATION, indices=True)):
    num_steps = len(set(data['word1']))

    for i, (word1, test_split) in enumerate(data.groupby('word1')):
        train_split = data[data['word1'] != word1]

        test_X = np.array(list(test_split['features']))
        test_Y = np.array(test_split['target'])

        banned_words1 = set(test_split['word1'])
        banned_words2 = set(test_split['word2'])

        word2_mask = train_split['word2'].map(lambda x: x not in banned_words2)
        #print np.sum(word1_mask), np.sum(word2_mask)
        train_view = train_split[word2_mask]
        train_X = np.array(list(train_view['features']))
        train_Y = np.array(train_view['target'])

        logging.debug("Learning %d/%d [%s training matrix]" % (i + 1, num_steps, train_X.shape))

        learned = klassifier.fit(train_X, train_Y)

        logging.debug("Testing '%s' %d/%d [%s testing matrix]" % (word1, i + 1, num_steps, test_X.shape))

        #probs = learned.predict_proba(test_X)
        labels = learned.predict(test_X)
        acc = np.sum(labels == test_Y) / float(len(labels))
        total_right += np.sum(labels == test_Y)
        total_total += len(labels)
        scores.append(acc)

        logging.debug("Processed row %3d/%3d (%2.1f); acc: %.3f; running acc: %.3f" % (i + 1, num_steps, 100. * (i + 1.) / NUM_CROSS_VALIDATION, acc, np.mean(scores)))

    logging.info("Done classifying!")
    logging.info("Classifier: %s" % klassifier)
    logging.info("Accuracy: %0.3f +/- %0.3f" % (np.mean(scores), 2 * np.std(scores)))
    logging.info("Accuracy: %0.3f " % (float(total_right) / total_total))

def findfeatures(data, klassifier, space, rmapper, num_components):
    train_X = np.array(list(data['features']))
    train_Y = np.array(data['target'])
    learned = klassifier.fit(train_X, train_Y)

    m = space.get_cooccurrence_matrix().get_mat()

    part1 = learned.coef_[:,:num_components]
    part2 = learned.coef_[:,num_components:]
    
    keepfeatures = (np.abs(part1) > 1e-3) | (np.abs(part2) > 1e-3)
    print np.sum(keepfeatures)
    import ipdb
    ipdb.set_trace()

    return

    #s = m.argsort(axis=0)

    NUM_SELECT = 250

    lookup = {i : x for i, (x, y) in enumerate([l.strip().split("\t") for l in open("/var/local/roller/data/dist-spaces/bigspace/vectorspace.cols")])}
    tm = space.operations[1]._DimensionalityReductionOperation__transmat.get_mat().todense()
    s = tm.argsort(axis=0)

    diff1 = tm[:,:num_components] * learned.coef_[:,:num_components].T
    diff2 = tm[:,:num_components] * learned.coef_[:,num_components:].T

    g1 = diff1.sum(axis=1).A.T[0]
    g2 = diff2.sum(axis=1).A.T[0]
    g3 = g1 + g2

    if TYPE == 'pos':
        keys1 = [lookup[y] for y in g1.argsort()[-NUM_SELECT:]]
        keys2 = [lookup[y] for y in g2.argsort()[-NUM_SELECT:]]
    elif TYPE == 'neg':
        keys1 = [lookup[y] for y in g1.argsort()[:NUM_SELECT]]
        keys2 = [lookup[y] for y in g2.argsort()[:NUM_SELECT]]
    elif TYPE == 'abs':
        keys1 = [lookup[y] for y in np.abs(g1).argsort()[-NUM_SELECT:]]
        keys2 = [lookup[y] for y in np.abs(g2).argsort()[-NUM_SELECT:]]
    for x in set(keys):
        print x

    for k, klass in rmapper.iteritems():
        break
        g1 = diff1[:,k].A.T[0]
        g2 = diff2[:,k].A.T[0]
        if TYPE == 'pos':
            keys1 = [lookup[y] for y in g1.argsort()[-NUM_SELECT:]]
            keys2 = [lookup[y] for y in g2.argsort()[-NUM_SELECT:]]
        elif TYPE == 'neg':
            keys1 = [lookup[y] for y in g1.argsort()[:NUM_SELECT]]
            keys2 = [lookup[y] for y in g2.argsort()[:NUM_SELECT]]
        elif TYPE == 'abs':
            keys1 = [lookup[y] for y in np.abs(g1).argsort()[-NUM_SELECT:]]
            keys2 = [lookup[y] for y in np.abs(g2).argsort()[-NUM_SELECT:]]
        for x in set(keys1 + keys2):
            print x
        print

    # for t in xrange(num_components):
    #     break
    #     wordweights = s[:,t].T.A[0]
    #     for x in reversed(np.concatenate([wordweights[:NUM_SHOW], wordweights[-NUM_SHOW:]])):
    #         print "       %2.3f    %5d   %s" % (tm[x,t], x, lookup[x])
    #     import ipdb
    #     ipdb.set_trace()
    # for c, row in enumerate(learned.coef_):
    #     klass = rmapper[c]
    #     print "Relevant features to '%s'-v-all classifier:" % klass
    #     most_rela = np.abs(row).argsort()[-NUM_SHOW:]
    #     print "    weight   topic"
    #     for j in reversed(most_rela):
    #         print "    %2.3f     %d" % (row[j], j)
    #         wordweights = s[:,j % num_components].T.A[0]
    #         import ipdb
    #         ipdb.set_trace()
    #         for i in xrange(NUM_SHOW):
    #             x = wordweights[-(i+1)]
    #             y = wordweights[i]
    #             print "       %-2.3f    %5d   %-15s         %-2.3f    %5d   %-10s" % (tm[x,j % num_components], x, lookup[x][:15], tm[y, j % num_components], y, lookup[y])





def main():
    parser = argparse.ArgumentParser(
                description='Classifies relations using a semantic space as features.')
    parser.add_argument('action', choices=('unseen', 'cv', 'findfeatures'), default='unseen',
                        help='Action to perform.')
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
    parser.add_argument('-p', '--predictions', action='store_true',
                        help='Output a CSV of model predictions')

    args = parser.parse_args()

    klassifier = classifier_factory(args.model)
    feature_generator = partial(generate_features, feattype=args.features, numfeatures=args.numfeatures)

    logging.info('Reading table...')
    data = pd.read_table(args.data, names=('word1', 'info', 'relation', 'word2'))
    #data = pd.read_table(args.data, names=('word1', 'word2', 'entails'))
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
    logging.info("%d pairs with features..." % len(data))

    if args.action == 'unseen':
        compute_unseen_accuracy(data, klassifier)
    elif args.action == 'cv':
        pass
    elif args.action == 'findfeatures':
        num_components = args.numfeatures and args.numfeatures or space.element_shape[0]
        findfeatures(data, klassifier, space, target_unmapper, num_components)
    elif args.action == 'train':
        pass
    else:
        raise ValueError("Invalid action, '%s'!" % args.action)


if __name__ == '__main__':
    main()


