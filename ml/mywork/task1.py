""" Features

The objective of this task is to explore the corpus, deals.txt. 

The deals.txt file is a collection of deal descriptions, separated by a new line, from which 
we want to glean the following insights:

1. What is the most popular term across all the deals?
2. What is the least popular term across all the deals?
3. How many types of guitars are mentioned across all the deals?

"""
import nltk
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

def wordCounts(raw_data):
	import sklearn.feature_extraction.text
	#ignore_words_list = open("ignore_words.txt").read().split()
	stopwords = nltk.corpus.stopwords.words('english')#+ignore_words_list
	tfidfv = sklearn.feature_extraction.text.TfidfVectorizer(stop_words = stopwords)
	tfidf_matrix = tfidfv.fit_transform(raw_data)
	freqs = [(word, tfidf_matrix.getcol(idx).sum()) for word, idx in tfidfv.vocabulary_.items()]
	sorted_freqs = sorted (freqs, key = lambda x: -x[1])
	return sorted_freqs

def typesOfGuitar(deals):
	from nltk import pos_tag
	guit_list = [sen for sen in deals if u"guitar" in sen.lower()]
	join_guit_list = " & ".join(guit_list)
	bigram_measures = nltk.collocations.BigramAssocMeasures()
	finder = BigramCollocationFinder.from_words(nltk.word_tokenize(join_guit_list))
	stopwords = nltk.corpus.stopwords.words('english')
	finder.apply_word_filter(lambda w: w in stopwords)
	finder.apply_freq_filter(2)
	scored = finder.score_ngrams(bigram_measures.raw_freq)
	sorted_scores = sorted(bigram for bigram, score in scored)
	join_sorted_scores = []
	types = set()
	for cur_tuple in sorted_scores:
		first_word = cur_tuple[0]
		second_word = cur_tuple[1]
		join_sorted_scores.append(first_word+" "+second_word)
	for word in join_sorted_scores:
		(first,second) = word.lower().split()
		first_tag, second_tag = pos_tag(word.lower().split())
		word, tag = first_tag
		if u'guitar' in second and tag in 'JJ':
			types.add(word)
	return (types, len(types))

def main():
	"""Retrieving the location of the file"""
	filename = "deals.txt"
	path = "../data/"+filename
	"""path describes where the corpus is located"""
	raw_data = open(path).readlines()
	tfidfWords = wordCounts(raw_data)
	max_term, max_count = tfidfWords[0]
	min_term, min_count = tfidfWords[-1]
	(guitar_types, num_guitar_types) = typesOfGuitar(raw_data)
	print "Most common term in the deals", max_term
	print "Least common term in the deals", min_term
	print guitar_types, num_guitar_types

if __name__ == '__main__':
	main()