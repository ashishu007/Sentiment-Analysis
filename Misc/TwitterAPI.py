import tweepy
from textblob import TextBlob

consumer_key = "zmLvww9CNLF1HJh6YmLuxLBXw"
consumer_secret = "iMRgZgRj7zpGKFYUhqrkikMafXNudFynMx6Gs09AkgPNzK1Jog"

access_token = "3562472131-dao7tDjYqwCgeKPLLDMd6E2Pw1bvOYdsNYN6aeW"
access_token_secret = "492ygQhoUZkIuEQlzZBDN0rdIMp1kPL6fckPC70iieYjT"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search("Narendra Modi")

ctr = 0
for tweet in public_tweets:
    ctr += 1
    print("************************ New Tweet ************************")
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)

print(ctr)