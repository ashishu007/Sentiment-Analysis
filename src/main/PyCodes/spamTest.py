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
# inpTweets = csv.reader(open(dircetory + 'Output\\csv\\NoConfidenceMotion.csv', 'r', encoding = "cp850"))
inpTweets = csv.reader(open(dircetory + 'Resources\\sms_spam_test.csv', 'r', encoding = "cp850"))
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
# print(featureList)

# print(tweets)
# print(extract_features("that class sucks, never attend it"))

testing_set = nltk.classify.util.apply_features(extract_features, tweets)
# print(testing_set)
# model_names = ["LogisticRegression_classifier", "MNB_classifier", "NaiveBayes_Classifier", \
#                 "SVC_classifier", "SGDClassifier_classifier", \
#                 "LinearSVC_classifier", "BernoulliNB_classifier"]

model_names = ["NaiveBayes_Classifier", "SVC_classifier", "LogisticRegression_classifier",
                "MNB_classifier", "SGDClassifier_classifier", "LinearSVC_classifier",
                "BernoulliNB_classifier"]

accuracy_list = []
for name in model_names:
    print("Now testing: " + name)
    classifier = pickle.load(open(dircetory + "Output\\Models\\Spam\\" + name + ".sav", 'rb'))
    accuracy_percentage = (nltk.classify.accuracy(classifier, testing_set))*100
    accuracy_list.append(accuracy_percentage)

print(accuracy_list)
dict1 = {
    "Algorithm": model_names,
    "Accuracy Percentage": accuracy_list
}

df = pd.DataFrame(dict1, columns=["Algorithm", "Accuracy Percentage"])
df.to_csv(dircetory + "Output\\csv\\SpamAccuracy.csv", index=0)
