""" Features

The objective of this task is to explore the corpus, deals.txt. 

The deals.txt file is a collection of deal descriptions, separated by a new line, from which 
we want to glean the following insights:

1. What is the most popular term across all the deals?
2. What is the least popular term across all the deals?
3. How many types of guitars are mentioned across all the deals?

"""


def wordCounts(filename, popularity):
	import sklearn.feature_extraction.text
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
	return sorted_freqs,sorted_freqs2
	#return "Real word", tfidf_word, "Unreal word", cv_word

def typesOFGuitar()
	"""
	Generate bigrams
	for all the indices where guitar occurs, get previos index and add the word if its not nonsense.
	"""

def main():
	filename = "deals.txt"
	(tfidfWords, cvWords) = wordCounts(filename)
	max_words = tfidfWords[0]
	min_words = tfidfWords[-1]
	
if __name__ == '__main__':
	main()