from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


text = open('query_input.txt', 'r').read().strip()
N = text[0]
sets = text[2:].split('*****')
A = sets[0].strip().split('\n')
B = sets[1].strip().split('\n')


vectorizer = TfidfVectorizer(
    ngram_range=(1, 5),
    stop_words='english',
    analyzer='char',
    max_df=0.5
)

vectorizer.fit(A+B)
A_vec = vectorizer.transform(A)
B_vec = vectorizer.transform(B)

max = 0
answer = None
for ind_a in range(len(A)):
    a_vec = A_vec[ind_a]
    for ind_b in range(len(B)):
        b_vec = B_vec[ind_b]
        sim = cosine_similarity(a_vec, b_vec).tolist()[0][0]
        if sim > max:
            max = sim
            answer = ind_b
    print(A[ind_a])
    print(B[answer])
    print('#####################')
    max = 0
    answer = None
