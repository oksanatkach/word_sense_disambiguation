import nltk
from math import log
import operator


text = open('query_input.txt', 'r').read().strip()
N = text[0]
sets = text[2:].split('*****')
A = sets[0].strip().split('\n')
B = sets[1].strip().split('\n')


def preproc(doc):
    doc = nltk.word_tokenize(doc)
    return doc


def tf(t, d):
    return d.count(t)


def idf(t, D):
    return log(float(len(D)) / len([d for d in D if t in d]))


def tf_idf(vocab, d, D):
    scores = []
    for t in vocab:
        scores.append(tf(t, d) * idf(t, D))
    return scores


def similarity(target, scores, vocab):
    scores.pop(' '.join(target))
    sims = {doc: 0 for doc in scores.keys()}
    for doc in scores:
        for word in target:
            ind = vocab.index(word)
            sims[doc] += scores[doc][ind]
    return max(sims.items(), key=operator.itemgetter(1))[0]


A = [preproc(doc) for doc in A]
B = [preproc(doc) for doc in B]

vocab = list(set([word for doc in A+B for word in doc]))
scores = {' '.join(d) : tf_idf(vocab, d, A+B) for d in A+B}


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

vec1 = np.array([scores[' '.join(A[0])]])
vec2 = np.array([scores[' '.join(A[1])]])

print(cosine_similarity(vec1, vec2))

max = 0
answer = ''
for a in A:
    vec1 = np.array([scores[' '.join(a)]])
    for b in B:
        vec2 = np.array([scores[' '.join(b)]])
        sim = cosine_similarity(vec1, vec2)
        if sim[0][0] > max:
            max = sim[0][0]
            answer = ' '.join(b)

    print(' '.join(a))
    print(answer)
    max = 0
    answer = ''

    # print(similarity(a, scores, vocab))
    print('#################')
