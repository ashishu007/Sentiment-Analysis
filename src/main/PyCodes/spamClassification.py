import re
import csv
import pprint
import nltk.classify
import pickle
import pandas as pd
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.svm import SVC, LinearSVC, NuSVC

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
#end

dircetory = "C:\\Users\\User\\Sentiment-Analysis\\src\\main\\"

#Read the tweets one by one and process it
inpTweets = csv.reader(open(dircetory + 'Resources\\sms_spam_train.csv', 'r', encoding = "cp850"))
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
# print("featureList", featureList)

# Generate the training set
training_set = nltk.classify.util.apply_features(extract_features, tweets)
# print("training set", training_set)

print("Train the Naive Bayes classifier")
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
print("Trained NaiveBayes_Classifier")

filename = 'NaiveBayes_Classifier.sav'
pickle.dump(NBClassifier, open(dircetory + "Output\\Models\\Spam\\" + filename, 'wb'))


print("Training SVC_classifier")
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("Trained SVC_classifier")

filename1 = 'SVC_classifier.sav'
pickle.dump(SVC_classifier, open(dircetory + "Output\\Models\\Spam\\" + filename1, 'wb'))


# print("Train the Max Entropy classifier")
# MaxEntClassifier = nltk.classify.maxent.MaxentClassifier.train(training_set, 'GIS', trace=3, \
#                     encoding=None, labels=None, gaussian_prior_sigma=0, max_iter = 10)
# print("ME trained")

# filename2 = 'Max_Entropy_new.sav'
# pickle.dump(MaxEntClassifier, open(dircetory + "Output\\Models\\" + filename2, 'wb'))


print("Training Logisitic Regression")
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("Trained Logisitic Regression")

filename3 = 'LogisticRegression_classifier.sav'
pickle.dump(LogisticRegression_classifier, open(dircetory + "Output\\Models\\Spam\\" + filename3, 'wb'))


print("Training MNB_classifier")
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("Trained MNB_classifier")

filename4 = 'MNB_classifier.sav'
pickle.dump(MNB_classifier, open(dircetory + "Output\\Models\\Spam\\" + filename4, 'wb'))


print("Training SGDClassifier_classifier")
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("Trained SGDClassifier_classifier")

filename5 = 'SGDClassifier_classifier.sav'
pickle.dump(SGDClassifier_classifier, open(dircetory + "Output\\Models\\Spam\\" + filename5, 'wb'))


print("Training LinearSVC_classifier")
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("Trained LinearSVC_classifier")

filename6 = 'LinearSVC_classifier.sav'
pickle.dump(LinearSVC_classifier, open(dircetory + "Output\\Models\\Spam\\" + filename6, 'wb'))


print("Training BernoulliNB_classifier")
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("Trained BernoulliNB_classifier")

filename7 = 'BernoulliNB_classifier.sav'
pickle.dump(BernoulliNB_classifier, open(dircetory + "Output\\Models\\Spam\\" + filename7, 'wb'))
