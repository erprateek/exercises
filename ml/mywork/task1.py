""" Features

The objective of this task is to explore the corpus, deals.txt. 

The deals.txt file is a collection of deal descriptions, separated by a new line, from which 
we want to glean the following insights:

1. What is the most popular term across all the deals?
2. What is the least popular term across all the deals?
3. How many types of guitars are mentioned across all the deals?

"""


def wordCounts(filename):
	import sklearn.feature_extraction.text
	import nltk
	raw_data = open(filename).readlines()
	cv = sklearn.feature_extraction.text.CountVectorizer()
	stopwords = nltk.corpus.stopwords.words('english')
	tfidfv = sklearn.feature_extraction.text.TfidfVectorizer(stop_words = stopwords)
	cv_matrix = cv.fit_transform(raw_data)
	tfidf_matrix = tfidfv.fit_transform(raw_data)
	freqs = [(word, tfidf_matrix.getcol(idx).sum()) for word, idx in tfidfv.vocabulary_.items()]
	freqs2 = [(word, cv_matrix.getcol(idx).sum()) for word, idx in cv.vocabulary_.items()]
	sorted_freqs = sorted (freqs, key = lambda x: -x[1])
	sorted_freqs2 = sorted (freqs2, key = lambda x: -x[1])
	"""
	Consider adding filter words since it is online deals so words such as Online, .com, link are avoided
	"""
	return sorted_freqs,sorted_freqs2
	#return "Real word", tfidf_word, "Unreal word", cv_word

def typesOfGuitar(filename, list_guitar_types):
	"""Generate bigrams
	for all the indices where guitar occurs, get previos index and add the word if its not not useful.
	"""
	"""Should generalize by the type of subject searched for
		Instead of hard coding guitar
	"""
	"""Remove the cv words category by the end"""
	"""More specific can be made by making it a 3 gram and analyzing what follows
		Should incorporate collocation in nltk. That might help in finding frequencies of the terms occurring more than once.
	"""
	"""should supply deals only containing guitars"""

	import sklearn.feature_extraction.text
	from nltk import pos_tag
	deals = open(filename).readlines()
	ngram_tfidf = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(2, 2))
	ngram_tfidf_matrix = ngram_tfidf.fit_transform(deals)
	bigrams = ngram_tfidf.vocabulary_.keys()
	types = set()
	for word in bigrams:
		if u"guitar" in word:
			[first,second] = word.split()
			if u"guitar" in second:
            	#print word
				first_tag,second_tag = pos_tag(word.split())
				cur_word, pos = first_tag
				if pos in ['JJ']:
					types.add(cur_word)
	if list_guitar_types:
		set_guitar_types = set(list_guitar_types)
		types.intersection_update(set_guitar_types)
	return (types, len(types))

def main():
	"""Retrieving the location of the file"""
	filename = "deals.txt"
	path = "../data/"+filename
	"""path describes where the corpus is located"""
	(tfidfWords, cvWords) = wordCounts(path)
	max_term, max_count = tfidfWords[0]
	min_term, min_count = tfidfWords[-1]
	(guitar_types, num_guitar_types) = typesOfGuitar(path, [])
	print "Most common term in the deals", max_term
	print "Least common term in the deals", min_term
	print guitar_types, num_guitar_types

if __name__ == '__main__':
	main()