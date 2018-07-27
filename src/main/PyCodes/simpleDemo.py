import re
import csv
import pprint
import nltk.classify
import pandas as pd

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

#Read the tweets one by one and process it
inpTweets = csv.reader(open('C:\\Users\\User\\Sentiment-Analysis\\src\\main\\Resources\\full_training_dataset.csv', 'r', encoding = "cp850"))
# inpTweets = pd.read_csv("data/full_training_dataset.csv", encoding="")
print(inpTweets)
stopWords = getStopWordList('C:\\Users\\User\\Sentiment-Analysis\\src\\main\\Resources\\stopwords.txt')
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

# Train the Naive Bayes classifier
print("Train the Naive Bayes classifier")
NBClassifier = nltk.NaiveBayesClassifier.train(training_set[:10])
print("train")

# Test the classifier
testTweet = 'This is very bad @ravikiranj, i heard you wrote a new tech post on sentiment analysis. That is ridculous'
processedTestTweet = processTweet(testTweet)
sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
print("testTweet = %s, sentiment = %s\n" % (testTweet, sentiment))

df = pd.read_csv("C:\\Users\\User\\Sentiment-Analysis\\src\\main\\Output\\GST.csv")
tweets = []
senti = []
for text in df["tweets"]:
    # print(text)
    if type(text) != str:
        continue
    processedTestTweet = processTweet(text)
    sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
    tweets.append(text)
    senti.append(sentiment)

dict1 = {
    "tweets": tweets,
    "senti": senti
}

df1 = pd.DataFrame(dict1, columns = ["tweets", "senti"])
df1.to_csv("C:\\Users\\User\\Sentiment-Analysis\\src\\main\\Output\\GST_new_test.csv")