""" Classification

The objective of this task is to build a classifier that can tell us whether a new, unseen deal 
requires a coupon code or not. 

We would like to see a couple of steps:

1. You should use bad_deals.txt and good_deals.txt as training data
2. You should use test_deals.txt to test your code
3. Each time you tune your code, please commit that change so we can see your tuning over time

Also, provide comments on:
    - How general is your classifier?
    - How did you test your classifier?

"""
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation


def classify(X_train, Y_train, X_test):
    """Classification step takes place here using NaiveBayes"""
    clf = GaussianNB()
    clf.fit(X_train.toarray(), Y_train)
    Y_pred = clf.predict(X_test.toarray())
    return Y_pred

def populate_datasets(files_list):
    """Organize data as X_train, Y_train, X_test"""
    good_deals = []
    bad_deals = []
    path = "../data/"
    for i in xrange(len(files_list)):
        fname = files_list[i]
        raw = open(path+fname).readlines()
        if("good" in fname):
            good_deals = raw
        elif("bad" in fname):
            bad_deals = raw
        elif("test" in fname):
            test_deals = raw
    good_features_labels = [(deal, "1") for deal in good_deals]
    bad_features_labels = [(deal, "0") for deal in bad_deals]

    train_deals = good_features_labels+bad_features_labels
    #train_set = good_features_labels+bad_features_labels
    X_train = feature_extractor(good_deals+bad_deals)
    bad_labels = [0]*len(bad_deals)
    good_labels = [1]*len(good_deals)
    import numpy as np
    Y_train = np.asarray(good_labels+bad_labels)
    X_test = feature_extractor(test_deals)
    return X_train, Y_train, X_test

def feature_extractor(data):
    """Using the TFIDF approach to getting features"""
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_features = 338, max_df=0.5,stop_words='english')
    mat = vectorizer.fit_transform(data)
    return mat

def crossvalidate(X_trn, Y_trn):
    """Cross validation with comparison to classifiers that classify as only good or only bad"""
    import numpy as np
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X_trn.toarray(), Y_trn, test_size=0.4, random_state=1)
    dumb_labels1 = Y_test.copy()
    dumb_labels2 = Y_test.copy()
    dumb_labels1[dumb_labels1 == 0] = 1;    #Labels all 1s
    dumb_labels2[dumb_labels2 == 1] = 0;    #Labels all 0s
    clf = GaussianNB()
    clf.fit(X_train, Y_train)
    accuracy = clf.score(X_test, Y_test)
    dumb_clf1_score = clf.score(X_test, dumb_labels1)
    dumb_clf2_score = clf.score(X_test, dumb_labels2)
    print "Classifier Score : ", accuracy
    print "Dumb_classifier with all 1s : ", dumb_clf1_score
    print "Dumb classifier with all 0s : ", dumb_clf2_score
    return accuracy

def main():
    filelist = ["good_deals.txt", "bad_deals.txt", "test_deals.txt"]
    X_train, Y_train, X_test = populate_datasets(filelist)
    corpus_file = "deals.txt"
    path = "../data/"
    corpus = open(path+corpus_file).readlines()
    test_labels = classify(X_train, Y_train, X_test)
    clf_score = crossvalidate(X_train, Y_train)

if __name__ == '__main__':
    main()