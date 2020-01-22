from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = open('query_input.txt', 'r').read().strip()

sets = text.split('\n*****\n')

titles = sets[0].split('\n')
N = int(titles[0])
titles = titles[1:]

descs = sets[1].split('\n')

vectorizer = TfidfVectorizer()

vectorizer.fit(titles + descs)

titles_vec = vectorizer.transform(titles)
descs_vec = vectorizer.transform(descs)

max = 0
answer = None

for ind in range(N):
    a_vec = titles_vec[ind]
    for ind_b in range(N):
        b_vec = descs_vec[ind_b]
        sim = cosine_similarity(a_vec, b_vec)
        if sim > max:
            max = sim
            answer = ind_b
    print(titles[ind])
    print(descs[answer])
    print('#################')
    max = 0
