# IPython log file

get_ipython().magic(u'cd "C:\\Users\\Prateek\\Documents\\GitHub\\exercises\\ml\\data"')
get_ipython().magic(u'logstart')
deals = open("deals.txt").readlines()
deals[0:10]
test_deals = deals[0:10]
from nltk import pos_tag
groups = set()
groups_list = []
for line in test_deals:
    words = line.lower().split()
    wordsntags = pos_tag(words)
    print wordsntags
    
for line in test_deals:
    words = line.lower().split()
    wordsntags = pos_tag(words)
    for (word,tag) in wordsntags:
        if tag in "NN":
            groups.add(word)
    groups_list.append(groups)
    groups = set()
    
groups_list
test_deals = deals[0:100]
groups = set()
groups_list = []
for line in test_deals:
    words = line.lower().split()
    wordsntags = pos_tag(words)
    for (word,tag) in wordsntags:
        if tag in "NN":
            groups.add(word)
    groups_list.append(groups)
    groups = set()
    
groups_list
for line in test_deals:
    words = line.lower().split()
    wordsntags = pos_tag(words)
    for (word,tag) in wordsntags:
        if tag in "PN":
            groups.add(word)
    groups_list.append(groups)
    groups = set()
    
for line in test_deals:
    words = line.lower().split()
    wordsntags = pos_tag(words)
    for (word,tag) in wordsntags:
        if tag in "NP":
            groups.add(word)
    groups_list.append(groups)
    groups = set()
    
groups_list = []
for line in test_deals:
    words = line.lower().split()
    wordsntags = pos_tag(words)
    for (word,tag) in wordsntags:
        if tag in "NP":
            groups.add(word)
    groups_list.append(groups)
    groups = set()
    
groups_list
for line in test_deals:
    words = line.split()
    wordsntags = pos_tag(words)
    for (word,tag) in wordsntags:
        if tag in "NP":
            groups.add(word)
    groups_list.append(groups)
    groups = set()
    
groups
groups_list
reader = CategorizedPlaintextCorpusReader('deals.txt')
import nltk
from nltk import CategorizedPlaintextCorpusReader
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
reader = CategorizedPlaintextCorpusReader('deals.txt')
reader = CategorizedPlaintextCorpusReader('../data/', r'deals.txt', cat_pattern=r'(.*)\.txt')
reader
len(reader.categories())
reader.categories()
from nltk.corpus import PlaintextCorpusReader
newcorpus = PlaintextCorpusReader('../data/','deals.txt')
newcorpus.raw
newcorpus.raw()
newcorpus.words()
len(newcorpus.words())
unique_words = set(newcorpus.words())
len(unique_words)
stopwords = nltk.corpus.stopwords.words('english')
set_stop = set(stopwords)
unique_words.difference_update(set_stop)
len(unique_words)
tags = pos_tag(unique_words)
tags = pos_tag(list(unique_words))
tags[0:10]
nouns_list = filter(word for (word, tag) in tags if tag in 'NN')
nouns_list = [word for (word, tag) in tags if tag in 'NN']
nouns_list
nouns_list[0:10]
from nltk.parse.generate import generate, generate_iter, demo_grammar
from nltk.parse.generate import generate, demo_grammar
from nltk.parse.generate import generate
from nltk import parse_cfg
from nltk.parse.generate import *
from nltk.parse.generate import grammar
grammar
print(grammar)
from sklearn.datasets import fetch_20newsgroups
data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)
data_train = fetch_20newsgroups(subset='train', categories=all,
                                shuffle=True, random_state=42,
                                remove=remove)
data_train = fetch_20newsgroups(subset='train', categories=all,
                                shuffle=True, random_state=42,
                                remove=())
sent = "get train travel throughout France, plus international TGV tickets to Switzerland, Italy, Germany, Belgium & Spain."
pos_tag(sent)
pos_tag(sent.split())
s = """International Shipping Available
FREE shipping  on orders over $75 - for a limited time! Enter code FABFREE13 at checkout.
giggle's Best Baby Registry mobile app
Stock up for baby with giggle's very own collection of baby essentials: giggle Better Basics! Every piece is the ultimate in comfort and quality.
best in stylish yet practical baby gear from giggle
get it wrapped: free gift wrapping, we'll make sure it looks great!!
giggle Exclusive Halo Sleepsacks
giggle Sale on clothing & accessories, limited time only!
organic baby products
Shop the best selection of fashion jewelry at Max & Chloe
Find great deals on The Dark Knight Rises statues and busts at ShopDCEntertainment.com! Shop now!
alli Weight Loss Aid
Link provides access to various electric vehicle standards relating to Rechargeable Energy Storage System (RESS), Plug-in Vehicles and more.
Link provides access to hundreds of thousands of standards.
Link provides access to techniques to implement and establish an effective IT security management system
Find great deals on The Dark Knight Rises film cells at ShopDCEntertainment.com! Shop now!
Free Shipping on Smittybilt Sure Step Side Bars
LaCie Porsche Design P'9220 1TB Mobile Hard drive SuperSpeed USB 3.0 - Refurbished (302000-R) 9345901
stylish baby furniture from giggle
best baby products from giggle
get train travel throughout France, plus international TGV tickets to Switzerland, Italy, Germany, Belgium & Spain.
Check out the best sales and markdowns on designer swimwear from SwimSpot.com!
Clearance Sale - Up to 50% On Cell Phone & Cell Phone Accessories & Worldwide Free Shipping
Order now and save 39% on The Dark Knight Trilogy Limited Edition Gift Set on Blu-ray and DVD from WBShop.com! A great gift for the holidays!
An international vacations will expand your horizons of culture and geography. Whether its a European adventure, Caribbean cruises, or exploring the Far East, Travelocity.ca can help find travel deals to top destinations worldwide.
Text Link - Paula Young Collection
This Get Satisfaction link, links to the Clickatell Get Satisfaction forum. Potential clients can interact with each other here and get answers to any questions they may have."""
s
sentences = s.split("\n")
sentences
nn_freq
nn_freq = dict()
for sent in sentences:
    cur_tags = pos_tag(sent.split())
    for (word,tg) in cur_tags:
        if("NN" == tg || "NNS" == tg):
            nn_freq[word] = nn_freq+1
            
for sent in sentences:
    cur_tags = pos_tag(sent.split())
    for (word,tg) in cur_tags:
        if("NN" == tg or "NNS" == tg):
            nn_freq[word] = nn_freq+1

nn_freq = defaultdict()
for sent in sentences:
    cur_tags = pos_tag(sent.split())
    for (word,tg) in cur_tags:
        if("NN" == tg or "NNS" == tg):
            if nn_freq[word]:
                nn_freq[word] = nn_freq+1
            else:
                nn_freq[word] = 1
                
nn_freq
for sent in sentences:
    cur_tags = pos_tag(sent.split())
    for (word,tg) in cur_tags:
        if("NN" == tg or "NNS" == tg):
            if nn_freq.has_key(word):
                nn_freq[word] = nn_freq+1
            else:
                nn_freq[word] = 1
                
for sent in sentences:
    cur_tags = pos_tag(sent.split())
    for (word,tg) in cur_tags:
        if("NN" == tg or "NNS" == tg):
            if nn_freq.has_key(word):
                nn_freq[word] = nn_freq[word]+1
            else:
                nn_freq[word] = 1
                
nn_freq
pos_tag(["we'll])
pos_tag(["we'll"])
for sent in sentences:
    cur_tags = pos_tag(sent.split())
    for (word,tg) in cur_tags:
        if("NN" == tg or "NNS" == tg):
            print word,tg
            if nn_freq.has_key(word):
                nn_freq[word] = nn_freq[word]+1
            else:
                nn_freq[word] = 1
                
pos_tag("(302000-R)")
pos_tag(["(302000-R)"])
pos_tag(["we'll"])
pos_tag("get it wrapped: free gift wrapping, we'll make sure it looks great!!".split())
"get it wrapped: free gift wrapping, we'll make sure it looks great!!".split()
pos_tag(["we'll"])
nltk.WordPunctTokenizer().tokenize("get it wrapped: free gift wrapping, we'll make sure it looks great!!")
sorted(nltk.WordPunctTokenizer().tokenize("get it wrapped: free gift wrapping, we'll make sure it looks great!!"))
sorted(nltk.WordPunctTokenizer().tokenize("get it wrapped: free gift wrapping, we'll make sure it looks great!!"), order=desc)
sorted(nltk.WordPunctTokenizer().tokenize("get it wrapped: free gift wrapping, we'll make sure it looks great!!"), reverse="True")
sorted(nltk.WordPunctTokenizer().tokenize("get it wrapped: free gift wrapping, we'll make sure it looks great!!"), reverse=True)
newcorpus.words()[0:10]
newcorpus.words()[550:560]
newcorpus.sents()
newcorpus.fileids()
newcorpus.readme()
newcorpus.readme
newcorpus.root()
newcorpus.root
f=open("deals.txt").readlines()
clean = open('deals.txt').read().replace('\n', ' ')
len(clean)
nltk.sent_tokenize(clean)
sen = nltk.sent_tokenize(clean)
words_list = [nltk.word_tokenize(sent) for sent in sen]
words_tagged = [nltk.pos_tag(words) for words in words_list]
from nltk.collocations import *
trigram_measures = nltk.collocations.TrigramAssocMeasures()
bigram_measures = nltk.collocations.BigramAssocMeasures()
join_f = " ".join(f)
finder = BigramCollocationFinder.from_words(nltk.word_tokenize(join_f))
stopwords = nltk.corpus.stopwords.words('english')
finder.apply_word_filter(lambda w: w in stopwords)
pos_tag(["Amazon"])
pos_tag(["amazon"])
def find_groups():
	corpus = open("..data/deals.txt").read().replace('\n', ' ')
	raw_data = open("..data/deals.txt").readlines()
	#sentences = nltk.sent_tokenize(corpus)
	#words_list = [nltk.word_tokenize(sent) for sent in sentences]
	#words_tags = [nltk.pos_tag(word_list) for word_list in words_list]
	stopwords = nltk.corpus.stopwords.words('english')
	tfidfv = sklearn.feature_extraction.text.TfidfVectorizer(stop_words = stopwords)
	tfidf_matrix = tfidfv.fit_transform(raw_data)
	freqs = [(word, tfidf_matrix.getcol(idx).sum()) for word, idx in tfidfv.vocabulary_.items()]
	filter_freqs = []
	for (word,freq) in freqs:
		(word,tag) = pos_tag(word)
		if "NN" in tag:
			filter_freqs.append((word,freq))
   
def find_groups():
	corpus = open("..data/deals.txt").read().replace('\n', ' ')
	raw_data = open("..data/deals.txt").readlines()
	#sentences = nltk.sent_tokenize(corpus)
	#words_list = [nltk.word_tokenize(sent) for sent in sentences]
	#words_tags = [nltk.pos_tag(word_list) for word_list in words_list]
	stopwords = nltk.corpus.stopwords.words('english')
	tfidfv = sklearn.feature_extraction.text.TfidfVectorizer(stop_words = stopwords)
	tfidf_matrix = tfidfv.fit_transform(raw_data)
	freqs = [(word, tfidf_matrix.getcol(idx).sum()) for word, idx in tfidfv.vocabulary_.items()]
	filter_freqs = []
	for (word,freq) in freqs:
		(word,tag) = pos_tag(word)
		if "NN" in tag:
			filter_freqs.append((word,freq))
   return filter_freqs
def find_groups():
	corpus = open("..data/deals.txt").read().replace('\n', ' ')
	raw_data = open("..data/deals.txt").readlines()
	#sentences = nltk.sent_tokenize(corpus)
	#words_list = [nltk.word_tokenize(sent) for sent in sentences]
	#words_tags = [nltk.pos_tag(word_list) for word_list in words_list]
	stopwords = nltk.corpus.stopwords.words('english')
	tfidfv = sklearn.feature_extraction.text.TfidfVectorizer(stop_words = stopwords)
	tfidf_matrix = tfidfv.fit_transform(raw_data)
	freqs = [(word, tfidf_matrix.getcol(idx).sum()) for word, idx in tfidfv.vocabulary_.items()]
	filter_freqs = []
	for (word,freq) in freqs:
		(word,tag) = pos_tag(word)
		if "NN" in tag:
			filter_freqs.append((word,freq))
	return filter_freqs

ff = find_groups()
def find_groups():
	corpus = open("deals.txt").read().replace('\n', ' ')
	raw_data = open("deals.txt").readlines()
	#sentences = nltk.sent_tokenize(corpus)
	#words_list = [nltk.word_tokenize(sent) for sent in sentences]
	#words_tags = [nltk.pos_tag(word_list) for word_list in words_list]
	stopwords = nltk.corpus.stopwords.words('english')
	tfidfv = sklearn.feature_extraction.text.TfidfVectorizer(stop_words = stopwords)
	tfidf_matrix = tfidfv.fit_transform(raw_data)
	freqs = [(word, tfidf_matrix.getcol(idx).sum()) for word, idx in tfidfv.vocabulary_.items()]
	filter_freqs = []
	for (word,freq) in freqs:
		(word,tag) = pos_tag(word)
		if "NN" in tag:
			filter_freqs.append((word,freq))
	return filter_freqs

ff = find_groups()
import sklearn.feature_extraction.text
ff = find_groups()
pos_tag(["Amazon"])
def find_groups():
	import sklearn.feature_extraction.text
	#corpus = open("..data/deals.txt").read().replace('\n', ' ')
	raw_data = open("..data/deals.txt").readlines()
	#sentences = nltk.sent_tokenize(corpus)
	#words_list = [nltk.word_tokenize(sent) for sent in sentences]
	#words_tags = [nltk.pos_tag(word_list) for word_list in words_list]
	stopwords = nltk.corpus.stopwords.words('english')
	tfidfv = sklearn.feature_extraction.text.TfidfVectorizer(stop_words = stopwords)
	tfidf_matrix = tfidfv.fit_transform(raw_data)
	freqs = [(word, tfidf_matrix.getcol(idx).sum()) for word, idx in tfidfv.vocabulary_.items()]
	filter_freqs = []
	for (word,freq) in freqs:
		[(word,tag)] = pos_tag([word])
		if "NN" in tag:
			filter_freqs.append((word,freq))
	return filter_freqs

ff = find_groups()
def find_groups():
	import sklearn.feature_extraction.text
	#corpus = open("..data/deals.txt").read().replace('\n', ' ')
	raw_data = open("deals.txt").readlines()
	#sentences = nltk.sent_tokenize(corpus)
	#words_list = [nltk.word_tokenize(sent) for sent in sentences]
	#words_tags = [nltk.pos_tag(word_list) for word_list in words_list]
	stopwords = nltk.corpus.stopwords.words('english')
	tfidfv = sklearn.feature_extraction.text.TfidfVectorizer(stop_words = stopwords)
	tfidf_matrix = tfidfv.fit_transform(raw_data)
	freqs = [(word, tfidf_matrix.getcol(idx).sum()) for word, idx in tfidfv.vocabulary_.items()]
	filter_freqs = []
	for (word,freq) in freqs:
		[(word,tag)] = pos_tag([word])
		if "NN" in tag:
			filter_freqs.append((word,freq))
	return filter_freqs

ff = find_groups()
len(ff)
ff[0:20]
sorted_freqs = sorted (ff, key = lambda x: -x[1])
sorted_freqs[0:20]
pos_tag(["xd"])
dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)
type(dataset.data)
dataset.data[0:10]
len(dataset.data)
dataset.data[0]
raw_data = open("deals.txt").read()
tfidfv = sklearn.feature_extraction.text.TfidfVectorizer(stop_words = stopwords)
tfidf_matrix = tfidfv.fit_transform(raw_data)
len(dataset.data)
len(ff)
type(ff)
ff[0:30]
sorted_freqs[0:30]
sorted_freqs[0:40]
sorted_freqs[0:50]
filter_sorted_freqs = [(word,freq) for (word,freq) in sorted_freqs if freq < 250.0]
filter_sorted_freqs[0:30]
filter_sorted_freqs_tags = [(word,pos_tag([word])) for (word,freq) in sorted_freqs if freq < 250.0]
filter_sorted_freqs_tags[0:30]
labels = dataset.target
len(labels)
labels[0]
labels[1]
labels[2]
import numpy as np
tk = np.unique(labels).shape[0]
tk
1/tk
tk
km = KMeans(n_clusters=tk, init='k-means++', max_iter=100, n_init=1)
from sklearn.cluster import KMeans, MiniBatchKMeans
km = KMeans(n_clusters=tk, init='k-means++', max_iter=100, n_init=1)
km = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)
km.fit(X)
X = tfidf_matrix
tfidf_matrix
type(tfidfv.get_feature_names())
len(f)
tfidf_matrix = tfidfv.fit_transform(f)
tfidf_matrix
type(tfidfv.get_feature_names())
len(tfidfv.get_feature_names())
a = 4L
a
format(4220963601, '4L')
format(4220963601, 'x')
int(4L)
tfidf_matrix.dtype
km = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)
km.fit(tfidf_matrix)
km.labels_
from sklearn import metrics
metrics.homogeneity_score(labels, km.labels_)
l = [1]
l**5
l*5
labels = l*59647
metrics.homogeneity_score(labels, km.labels_)
metrics.completeness_score(labels, km.labels_)
metrics.adjusted_rand_score(labels, km.labels_))
metrics.adjusted_rand_score(labels, km.labels_)
import math
math.sqrt(59647/2)
tfidf_matrix.shape()
tfidf_matrix.shape
len(tfidfv.vocabulary_)
len(f)
tfidf_matrix[0,0]
tfidf_matrix[0,1]
tfidf_matrix[0]
len(filter_sorted_freqs)
filter_sorted_freqs[25640:25648]
tfidf_matrix[0]
tfidf_matrix.[0]
tfidf_matrix[0]
indices = np.nonzero(tfidf_matrix[0])
indices
indices = np.nonzero(tfidf_matrix[0][1])
indices = np.nonzero(tfidf_matrix[0][0])
indices = np.nonzero(tfidf_matrix[0])[0]
indices
indices = np.nonzero(tfidf_matrix[0])[1]
indices
sorted_freqs[0:50]
ignore_words_list = "online com free shipping link shop save sale deals buy code coupons".split()
"coupon" in ignore_words_list
for i in xrange(len(ignore_words_list)):
    if "coupon" in ignore_words_list[i]:
        print ignore_words_list[i]
        
sorted_freqs[50:100]
sorted_freqs[100:150]
sorted_freqs[150:200]
len(sorted_freqs)
import nltk.cluster.gaac
from nltk.cluster.gaac import *
gc = GAAClusterer(5)
gc.cluster(tfidf_matrix)
alist = list(numpy.array(tfidf_matrix).reshape(-1,))
gc.cluster(alist)
tdtd = tfidf_matrix.toarray()
tdtd
gc.cluster(tdtd)
from sklearn.decomposition import TruncatedSVD
lsa = TruncatedSVD()
tfidf_matrix_2 = lsa.fit_transform(tfidf_matrix)
tfidf_matrix_2 = Normalizer(copy=False).fit_transform(tfidf_matrix_2)
from sklearn.preprocessing import Normalizer
tfidf_matrix_2 = Normalizer(copy=False).fit_transform(tfidf_matrix_2)
tfidf_matrix_2
gc.cluster(tfidf_matrix_2)
