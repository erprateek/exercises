""" Groups and Topics

The objective of this task is to explore the structure of the deals.txt file. 

Building on task 1, we now want to start to understand the relationships to help us understand:

1. What groups exist within the deals?
2. What topics exist within the deals?

"""

import nltk
import sklearn.feature_extraction.text
from sklearn import cluster
from collections import Counter
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from task1 import wordCounts
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
import itertools

def find_groups():
    raw_data = open("../data/deals.txt").readlines()
    print "Getting words to be ignored from ignore_words.txt"
    stopwords = populateStopWords()
    print "Vectorizing Data using a TFIDF vectorizer"
    tfidfv = sklearn.feature_extraction.text.TfidfVectorizer(stop_words = stopwords)
    tfidf_matrix = tfidfv.fit_transform(raw_data)
    print "Performing dimensionality reduction by Singular Value Decomposition to 100 features"
    X = performSVD(tfidf_matrix, 100)
    print "Clustering deals using Kmeans clustering into k=sqrt(n/2) = 170 clusters"
    cls = cluster.KMeans(n_clusters=170)
    cls.fit(X)
    values = cls.cluster_centers_.squeeze()
    labels = cls.labels_
    print "Performing post processing steps"
    deal_mapping_tagged, deal_words = tfidf_info(tfidf_matrix, tfidfv)
    deal_words_list = deal_words.values()
    deal_groups = zip(labels, deal_mapping_tagged.values())
    de = list(accumulate(deal_groups))
    print "Getting commonly used words in English"
    (usual,unusual) = usualAndUnusualWords(tfidfv.vocabulary_.keys())
    d_label_words = {}
    for (x,y) in de:
        if d_label_words.has_key(x):
            d_label_words[x] = d_label_words[x]+y
        else:
            d_label_words[x] = y
    d_label_words_count = {}
    print "Performing labelling"
    for k,v in d_label_words.items():
         """Considering only those words which have synonyms i.e. filtering only recognized words in english"""
         vs = [word for sblist in v for (word,tag) in sblist if (word not in stopwords and len(wn.synsets(word))>0)]
         cnt = Counter(vs)
         d_label_words_count[k] = cnt.most_common(20)
    D_labl_words_cnt = d_label_words_count.values()
    # D_labl_words_cnt contains 170 clusters with each cluster containing the most common 20 words with their counts
    labellist = map(labllist, D_labl_words_cnt)
    # Assigning topics according to the non-max scores in clusters
    pre_topics_list = map(topicslist, D_labl_words_cnt)
    topics = topic_label_list(pre_topics_list,labellist)
    topics_labels = tuple_with_list(topics)
    # Can filter more through the following two lines
    #relevant_topics = filter(lambda (x,y): len(y)>1, topics)
    #topics_list = list(set([topic for topic,labels in relevant_topics]))
    topics_list = list(set([topic for topic,labels in topics]))
    group_names = list(set([group[0] for group in labellist if len(group)>0]))
    print "Finished"
    return group_names, topics_list

def tuple_with_list(topics):
    d = {}
    for (topic,label) in topics:
        d.setdefault(topic, []).append(label)
    for key, value in d.items():
        d[key] = list(set(d[key]))
    return d.items()

def wordContext(deals, search):
    guit_list = [sen for sen in deals if search in sen.lower()]
    join_guit_list = " ".join(guit_list)
    print "Calculating Bigram associations"
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(nltk.word_tokenize(join_guit_list))
    stopwords = populateStopWords()
    finder.apply_word_filter(lambda w: w in stopwords)
    finder.apply_freq_filter(2)
    scored = finder.score_ngrams(bigram_measures.raw_freq)
    print "Sorting bigrams"
    sorted_scores = sorted(bigram for bigram, score in scored)
    join_sorted_scores = []
    word_context = set()
    for word in sorted_scores:
        (first,second) = word
        if search in second:
            word_context.add(first)
        else:
            word_context.add(second)
    return (search, list(word_context))

def populateStopWords():
    ignore_words_list = open("ignore_words.txt").read().split()
    ignore_words = ignoreWordsList(ignore_words_list)
    stopwords = nltk.corpus.stopwords.words('english')+ignore_words
    return stopwords

def labllist(li):
    """Perform labelling based on the counts of the labels"""
    wrds,cts = zip(*li)
    max_score = max(cts)
    labels = [wrds[i] for i in xrange(len(cts)) if cts[i]==max_score]
    return labels

def topicslist(li):
    """Perform labelling based on the counts of the labels"""
    wrds,cts = zip(*li)
    max_score = max(cts)
    topics = [wrds[i] for i in xrange(len(cts)) if cts[i]<max_score]
    return topics

def topic_label_list(tpc,lbl):
    """Reverse mapping topics to labels from labels to topics"""
    li = []
    for i in xrange(len(tpc)):
        labels = lbl[i]
        topics = tpc[i]
        for j in xrange(len(topics)):
            li.append((topics[j],labels[0])) 
    li = list(set(li))
    return li

def performSVD(tfidf_matrix, n):
    """Singular Value Decomposition on the matrix to reduce dimensionality to n features"""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import Normalizer
    svd = TruncatedSVD(n_components = n)
    X = svd.fit_transform(tfidf_matrix)
    X = Normalizer(copy=False).fit_transform(X)
    return X

def accumulate(l):
    """Label-wise binning the deals
    Yields label, [deals that are a part of the label]
    """
    import itertools
    import operator
    it = itertools.groupby(l, operator.itemgetter(0))
    for key, subiter in it:
        yield key, [item[1] for item in subiter]

def tfidf_info(tfidf_matrix, tfidfv):
    """Computing information from the tfidf vector and the matrix
    Returns 2 dictionaries, D2 and D4. D4 contains the deal numbers as key and 
    tags of the words as values if they are nouns.
    """
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
        D4[key] = [(word, words_n_tags[word]) for word in value if "NN" in words_n_tags[word]]
    return D4, D2

def usualAndUnusualWords(words):
    """Getting the englist words 
    This helps remove the non english words, typos, words which have no english meaning
    """
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    text_vocab = set(w.lower() for w in words if w.isalpha())
    unusual = text_vocab.difference(english_vocab)
    usual = text_vocab.intersection(english_vocab)
    return (usual, unusual)

def ignoreWordsList(ignore_words_list):
    """Ignoring words considering synonyms of the words"""
    from nltk.corpus import wordnet as wn
    syns = []
    for word in ignore_words_list:
        li = wn.synsets(word)
        for ss in li:
            syns.append(ss.lemma_names)
    all_ignore = []
    for wlist in syns:
        all_ignore.extend(wlist)
    all_ignore = all_ignore+ignore_words_list
    all_ignore = list(set(all_ignore))
    return all_ignore