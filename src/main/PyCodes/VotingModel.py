import nltk
import random
import re
import csv
import pprint
import nltk.classify
import pickle
import pandas as pd
from nltk.corpus import movie_reviews
from statistics import mode
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.svm import SVC, LinearSVC
from nltk.classify import ClassifierI
# from collections import Counter


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        # temp = Counter(votes)
        # res = temp.most_common(1)
        # return res
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

dircetory = "C:\\Users\\User\\Sentiment-Analysis\\src\\main\\"

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL) 
    return pattern.sub(r"\1\1", s)
#end

#start process_tweet
def processTweet(tweet):
    # process the tweets
    
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)    
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end 

#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

#start getfeatureVector
def getFeatureVector(tweet, stopWords):
    featureVector = []  
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences 
        w = replaceTwoOrMore(w) 
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if it consists of only words
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        #ignore if it is a stopWord
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector    
#end

#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features

#Read the tweets one by one and process it
inpTweets = csv.reader(open(dircetory + 'Resources\\testing_final.csv', 'r', encoding = "cp850"))
# inpTweets = pd.read_csv("data/full_training_dataset.csv", encoding="")
print(inpTweets)
stopWords = getStopWordList(dircetory + 'Resources\\stopwords.txt')
count = 0
featureList = []
tweets = []

for row in inpTweets:
    # print(row)
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment))
#end loop

# Remove featureList duplicates
featureList = list(set(featureList))
testing_set = nltk.classify.util.apply_features(extract_features, tweets)

SVC_classifier = pickle.load(open(dircetory + "Output\\Models\\SVC_classifier.sav", 'rb'))
NaiveBayes_Classifier = pickle.load(open(dircetory + "Output\\Models\\NaiveBayes_Classifier.sav", 'rb'))
LinearSVC_classifier = pickle.load(open(dircetory + "Output\\Models\\LinearSVC_classifier.sav", 'rb'))
SGDClassifier_classifier = pickle.load(open(dircetory + "Output\\Models\\SGDClassifier_classifier.sav", 'rb'))
MNB_classifier = pickle.load(open(dircetory + "Output\\Models\\MNB_classifier.sav", 'rb'))
BernoulliNB_classifier = pickle.load(open(dircetory + "Output\\Models\\BernoulliNB_classifier.sav", 'rb'))
LogisticRegression_classifier = pickle.load(open(dircetory + "Output\\Models\\LogisticRegression_classifier.sav", 'rb'))

voted_classifier = VoteClassifier(
                                # NaiveBayes_Classifier,
                                LinearSVC_classifier,
                                # SGDClassifier_classifier,
                                # MNB_classifier,
                                # BernoulliNB_classifier,
                                # SVC_classifier,
                                # LogisticRegression_classifier
                                )

print("Let's see the result of voting classifier")
result = (nltk.classify.accuracy(voted_classifier, testing_set))*100
print("voted_classifier accuracy percent:", result)
