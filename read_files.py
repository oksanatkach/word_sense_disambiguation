from nltk.tokenize import sent_tokenize, word_tokenize
import re
import itertools
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import neighbors
import numpy as np
from math import log

punctuation = ["'",',','!','?','{','}','[',']','(',')','.','``',"''",':',';','`']
window = 10
comp = 'data/apple-computers.txt'
fruit = 'data/apple-fruit.txt'


def open_file(path):
    fh = open(path, 'r')
    text = fh.read()
    text = text.replace('\t', ' ')
    text = re.sub('\[.*?\]', '', text)
    return text.split('\n')


def get_contexts(pars):
    contexts = []
    for par in pars:
        if 'apple' in par.lower():
            words = word_tokenize(par)
            apple_inds = [ ind for ind, word in enumerate(words) if 'apple' in word.lower() ]
            for ind in apple_inds:
                left_context = calc_context(words[:ind], 'left')
                right_context = calc_context(words[ind+1:], 'right')
                contexts.append(left_context + right_context)
    return contexts


def calc_context(words, side):

    def add_token(token, context, i):
        if token.isdigit():
            context.append('<NUM>')
            i += 1
        else:
            if token not in punctuation:
                if token not in ["''", "``"]:
                    context.append(token)
                    i += 1
        return context, i

    context = []

    if len(words) < window:
        for token in words:
            context, _ = add_token(token, context, 0)
    else:
        if side == 'left':
            words = list(reversed(words))
        i = 0
        for token in words:
            if i < window:
                context, i = add_token(token, context, i)
    if side == 'left':
        context = list(reversed(context))
    return context


def PPMI(contexts, vocab, pars):

    ppmi = { word: 0 for word in vocab }
    corpus = word_tokenize('\n'.join(pars))
    corpus_wc = len(corpus)
    apple_prob = len(contexts) / corpus_wc

    for word in vocab:
        mutual_count = 0
        for cont in contexts:
            if word in cont:
                mutual_count += 1

        if mutual_count > 0:
            mutual_prob = mutual_count / corpus_wc
            word_prob = corpus.count(word)/corpus_wc

            if word_prob > 0:
                pmi = log(mutual_prob / (word_prob * apple_prob), 2 )

            if pmi > 0:
                ppmi[word] = pmi
    return ppmi


def vectorize(contexts, vocab, ppmi):
    vectors = []
    for context in contexts:
        vector = []
        for word in vocab:
            if word in context:
                vector.append(ppmi[word])
            else:
                vector.append(0)
        vectors.append(vector)
    return vectors


def prep_test(data, vocab, word_count):
    contexts = get_contexts(data.split('\n'))
    vector = vectorize(contexts, vocab, word_count)
    return np.array(vector)


def train(X_train, y_train):
    svm_clf = svm.LinearSVC(max_iter=300000)
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=20)

    svm_clf.fit(X_train, y_train)
    knn_clf.fit(X_train, y_train)

    return svm_clf, knn_clf


def test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    errors = 0
    for ind in range(len(y_pred)):
        if y_pred[ind] != y_test[ind]:
            errors += 1
    return errors


if __name__ == "__main__":
    comp_pars = open_file(comp)
    fruit_pars = open_file(fruit)

    comp_contexts = get_contexts(comp_pars)
    fruit_contexts = get_contexts(fruit_pars)

    ALL = list(itertools.chain.from_iterable(comp_contexts + fruit_contexts))
    vocab = list(set(ALL))
    word_count = Counter(ALL)

    comp_ppmi = PPMI(comp_contexts, vocab, comp_pars+fruit_pars)
    fruit_ppmi = PPMI(fruit_contexts, vocab, comp_pars+fruit_pars)
    print(fruit_ppmi)

    comp_vectors = vectorize(comp_contexts, vocab, comp_ppmi)
    comp_labels = [0] * len(comp_vectors)
    fruit_vectors = vectorize(fruit_contexts, vocab, fruit_ppmi)
    fruit_labels = [1] * len(fruit_vectors)

    X = np.array(comp_vectors + fruit_vectors)
    y = np.array(comp_labels + fruit_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    svm_clf, knn_clf = train(X_train, y_train)

    svm_errors = test(svm_clf, X_test, y_test)
    knn_errors = test(knn_clf, X_test, y_test)

    print(len(y_test))
    print(svm_errors)
    print(knn_errors)

    # test_sent = "Apple has become the world's first public company to be worth $1 trillion (£767bn)."
    test_sent = "Photooxidative and heat stress stimulate sunburn development on apple fruit in the field growing under increasingly stressful conditions."
    test_vector1 = prep_test(test_sent, vocab, comp_ppmi)
    test_vector = prep_test(test_sent, vocab, fruit_ppmi)

    print(svm_clf.predict(test_vector))
    print(svm_clf.predict(test_vector1))

    print(knn_clf.predict(test_vector))
    print(knn_clf.predict(test_vector1))
