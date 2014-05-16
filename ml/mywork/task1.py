""" Features

The objective of this task is to explore the corpus, deals.txt. 

The deals.txt file is a collection of deal descriptions, separated by a new line, from which 
we want to glean the following insights:

1. What is the most popular term across all the deals?
2. What is the least popular term across all the deals?
3. How many types of guitars are mentioned across all the deals?

"""

### SOLUTION - TASK 1

# For getting the word counts, we remove the stopwords since they are 
# meaningless. This stopwords list populated using the most common terms 
# occurring in the deals without any metric and adding to the list of 
# common english words.
# I also have filtered all the numbers when finding the most frequent and 
# least frequent terms in the file.

import nltk
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from task2 import populateStopWords, ignoreWordsList

def wordCounts(raw_data):
	"""Computing the word counts here to get the most and the 
	least frequent words"""
	import sklearn.feature_extraction.text
	stopwords = populateStopWords()
	# Vectorizing the data
	print "Vectorizing"
	tfidfv = sklearn.feature_extraction.text.TfidfVectorizer(stop_words = stopwords)
	tfidf_matrix = tfidfv.fit_transform(raw_data)
	print "Post processing"
	freqs = [(word, tfidf_matrix.getcol(idx).sum()) for word, idx in tfidfv.vocabulary_.items()]
	sorted_freqs = sorted (freqs, key = lambda x: -x[1])
	# Filtering to remove numbers that might be occurring
	# Similar filter can be used to find the most common discount percentages
	sorted_list = [word for word,score in sorted_freqs if word.isalpha()]
	return sorted_list

def typesOfObject(deals, obj):
	"""The method takes in raw text and determines the types of Guitars
	found in the text"""
	from nltk import pos_tag
	# Filter deals only that contain the required object
	guit_list = [sen for sen in deals if obj in sen.lower()]
	join_guit_list = " & ".join(guit_list)
	print "Computing Bigram association metrics"
	guitar_bigrams = BigramMetrics(join_guit_list, obj)
	types = set()
	# Looking for adjective associations of guitars that describe the guitars
	for word in guitar_bigrams:
		(first,second) = word.lower().split()
		first_tag, second_tag = pos_tag(word.lower().split())
		word, tag = first_tag
		if obj in second and tag in 'JJ':
			types.add(word)
	return (types, len(types))

def BigramMetrics(word_list, obj):
	"""Considers the bigram associations in the sentences
	that contain the object, and computes their scores based 
	on their occurrence"""
	bigram_measures = nltk.collocations.BigramAssocMeasures()
	finder = BigramCollocationFinder.from_words(nltk.word_tokenize(word_list))
	# Removing the stop words
	"""Test once with populatestopwords also"""
	stopwords = nltk.corpus.stopwords.words('english')
	finder.apply_word_filter(lambda w: w in stopwords)
	finder.apply_freq_filter(2)
	scored = finder.score_ngrams(bigram_measures.raw_freq)
	sorted_scores = sorted(bigram for bigram, score in scored)
	join_sorted_scores = [cur_tuple[0]+" "+cur_tuple[1] for cur_tuple in sorted_scores if obj in cur_tuple[1].lower()]
	guitar_scores = filter(lambda word: obj in word.lower(), join_sorted_scores)
	return guitar_scores

def main():
	"""Retrieving the location of the file"""
	filename = "deals.txt"
	path = "../data/"+filename
	"""path describes where the corpus is located"""
	raw_data = open(path).readlines()
	print "Computing Word Counts using TF-IDF"
	tfidfWords = wordCounts(raw_data)
	max_term = tfidfWords[0]
	min_term = tfidfWords[-1]
	obj = u"guitar"	#Since all comparisons are done in lower, specify unicode lowercase string
	print "Getting the types of the object ", obj 
	(guitar_types, num_guitar_types) = typesOfObject(raw_data, obj)
	print "Most common term in the deals ", max_term
	print "Least common term in the deals ", min_term
	print guitar_types, num_guitar_types

if __name__ == '__main__':
	main()