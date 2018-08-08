#!/usr/bin/python
import tweepy
import re
import pandas as pd

consumer_key = "JiNb0JkXyIy7sh5B9QZa1DSSu"
consumer_secret = "OYvmtgQEAveN7QrelH9mwvftXZa7cBBLM2GORh3gaEDl61EL8H"

access_token = "3562472131-F18yEVO6pahXndKwhDEAZ6y9J0wphG6ih0wYzOe"
access_token_secret = "i73BrAzKBHquAKlEHrC9QtBjH4Z6GHHnWEFdLkU2LDi6o"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

time = []
tweets = []
ctr = 0 

dircetory = "C:\\Users\\User\\Sentiment-Analysis\\src\\main\\"

for tweet in tweepy.Cursor(api.search,q="#NoConfidenceMotion",count=100,
                           lang="en",
                           since="2018-07-15").items(4000):
    time.append(tweet.created_at)
    temp = tweet.text
    temp1 = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', 'URL', temp)
    tweets.append(temp1)
    ctr += 1
    print(ctr)

raw_data = {
    "time": time,
    "tweets": tweets
}
df = pd.DataFrame(raw_data, columns = ["time", "tweets"])
df.to_csv(dircetory + "Output\\csv\\NoConfidenceMotion.csv", index=0)