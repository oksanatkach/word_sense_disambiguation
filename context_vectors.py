import re
from nltk.tokenize import word_tokenize
from string import punctuation
import numpy as np
from sklearn import svm
from sklearn import neighbors

comp_path = 'data/apple-computers.txt'
fruit_path = 'data/apple-fruit.txt'
window = 10


def read_file(path):
    text = open(path, 'r').read().strip()
    text = re.sub( '\[\d*?\]', '', text )
    text = text.replace('\t', ' ')
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
                # print(len(left_context + right_context))
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


def word_counts(ALL, vocab):

    wcs = {}

    for word in vocab:
        wcs[word] = ALL.count(word)

    return wcs


def vectorize(contexts, vocab, word_counts):
    vectors = []

    for context in contexts:
        vector = []
        for word in vocab:
            if word in context:
                vector.append(word_counts[word])
            else:
                vector.append(0)
        vectors.append(vector)

    return vectors


def train(X_train, y_train):
    svm_clf = svm.LinearSVC(max_iter=300000)
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=50)

    svm_clf.fit(X_train, y_train)
    knn_clf.fit(X_train, y_train)

    return svm_clf, knn_clf


def test(model, X_test, y_test):
    fp = 0
    for ind in range(len(X_test)):
        vector = X_test[ind]
        y_true = y_test[ind]
        y_pred = model.predict([vector])[0]
        if y_true != y_pred:
            fp += 1
    return fp


comp_par = read_file(comp_path)
fruit_par = read_file(fruit_path)

comp_contexts = get_contexts(comp_par)
fruit_contexts = get_contexts(fruit_par)
contexts = comp_contexts + fruit_contexts

ALL = [ el for lst in contexts for el in lst ]
vocab = set(ALL)

word_counts = word_counts(ALL, vocab)
print(word_counts)

ppmi_comp = {}
ppmi_fruit = {}


# comp_vectors = vectorize(comp_contexts, vocab, word_counts)
# comp_labels = [0] * len(comp_contexts)
#
# fruit_vectors = vectorize(fruit_contexts, vocab, word_counts)
# fruit_labels = [1] * len(fruit_contexts)
#
# from sklearn.model_selection import train_test_split
#
# X = np.array(comp_vectors + fruit_vectors)
# y = np.array(comp_labels + fruit_labels)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#
# svm_clf, knn_clf = train(X_train, y_train)

# new_example = 'I just bought a new Apple computer and I am very satisfied.'
# new_words = word_tokenize(new_example)
# new_vector = np.array(vectorize([new_words], vocab, word_counts))
# print(new_vector)

# print(svm_clf.predict(new_vector))
# print(len(X_test))
# print(test(svm_clf, X_test, y_test))
# print(test(knn_clf, X_test, y_test))
