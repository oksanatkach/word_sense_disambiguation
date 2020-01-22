import gensim


DIR = 'GoogleNews-vectors-negative300.bin'

model = gensim.models.KeyedVectors.load_word2vec_format(DIR, binary=True)
model.init_sims(replace=True)

print(model.wv['queen'])

