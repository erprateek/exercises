""" Groups and Topics

The objective of this task is to explore the structure of the deals.txt file. 

Building on task 1, we now want to start to understand the relationships to help us understand:

1. What groups exist within the deals?
2. What topics exist within the deals?

"""
import nltk
from nltk import pos_tag

def find_groups():
    import sklearn.feature_extraction.text
    raw_data = open("../data/deals.txt").readlines()
    ignore_words_list = open("ignore_words.txt").read().split()
    ignore_words = ignoreWordsList(ignore_words_list)
    stopwords = nltk.corpus.stopwords.words('english')+ignore_words
    tfidfv = sklearn.feature_extraction.text.TfidfVectorizer(stop_words = stopwords)
    tfidf_matrix = tfidfv.fit_transform(raw_data)
    freqs = [(word, tfidf_matrix.getcol(idx).sum()) for word, idx in tfidfv.vocabulary_.items()]
    (usual,unusual) = usualAndUnusualWords(tfidfv.vocabulary_.keys())
    word_scores = dict(freqs)
    deal_mapping_tagged, deal_words = tfidf_info(tfidf_matrix, tfidfv)
    num_deals = len(deal_mapping_tagged.keys())
    label_list = []
    deal_words_list = deal_words.values()
    scores = [[word_scores[word] for word in d_words] for d_words in deal_words_list]
    max_scores = [max(score) for score in scores]
    max_indices = [score.index(max_scores[idx]) for idx, score in enumerate(scores)]
#    max_indices = [score.index(max_score) for score in scores for max_score in max_scores]
    labels = [deal_words_list[deal][max_index] for deal, max_index in enumerate(max_indices)]
    unique_labels = set(labels).intersection(usual)

def performSVD(tfidf_matrix):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import Normalizer
    svd = TruncatedSVD(n_components = 100)
    X = svd.fit_transform(tfidf_matrix)
    X = Normalizer(copy=False).fit_transform(X)
    return X

def accumulate(l):
    import itertools
    import operator
    it = itertools.groupby(l, operator.itemgetter(0))
    for key, subiter in it:
        yield key, [item[1] for item in subiter]

def tfidf_info(tfidf_matrix, tfidfv):
    nz = tfidf_matrix.nonzero()
    (tr,tc) = nz
    tr = tr.tolist()
    tc = tc.tolist()
    de = list(accumulate(zip(tr,tc)))
    D = {}
    for (x,y) in de:
        D[x] = y
    D2 = {}
    inv_map = {v:k for k, v in tfidfv.vocabulary_.items()}
    for key in D.keys():
        D2[key] = [inv_map[k] for k in D[key]]
    words_n_tags = {}
    for key in tfidfv.vocabulary_.keys():
        [(word,tag)] = pos_tag([key])
        words_n_tags[key] = tag
    D4 = {}
    for key, value in D2.items():
        D4[key] = [(word, words_n_tags[word]) for word in value]
    return D4, D2

def usualAndUnusualWords(words):
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    text_vocab = set(w.lower() for w in words if w.isalpha())
    unusual = text_vocab.difference(english_vocab)
    usual = text_vocab.intersection(english_vocab)
    return (usual, unusual)

def ignoreWordsList(ignore_words_list):
    from nltk.corpus import wordnet as wn
    syns = []
    for word in ignore_words_list:
        li = wn.synsets(word)
        for ss in li:
            syns.append(ss.lemma_names)
    #syns = "".join(syns)
    all_ignore = []
    for wlist in syns:
        all_ignore.extend(wlist)
    all_ignore = all_ignore+ignore_words_list
    all_ignore = list(set(all_ignore))
    return all_ignore