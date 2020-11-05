#created by jhavedant
#under the mentorship of Dr. Shefalika Ghosh Samaddar
#importing all required libraries

import tweepy  # To consume Twitter's API
import pandas as pd  # To handle data
import numpy as np  # For number computing

# For plotting and visualization:
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
from displayfunction import display


# consumer key, consumer secret, access token, access secret. , for using the tweepy api ,
# to get the consumer key goto apps.twitter.com

ckey = ""
csecret = ""
atoken = ""
asecret = ""

# We import our access keys:
from credentials import *  # This will allow us to use the keys as variables


# API's setup:
def twitter_setup():
    """
    Utility function to setup the Twitter's API
    with our access keys provided.
    """
    # Authentication and access using keys:
    auth = tweepy.OAuthHandler(ckey, csecret)
    auth.set_access_token(atoken, asecret)

    # Return API with authentication:
    api = tweepy.API(auth)
    return api


extractor = twitter_setup()

# We create a tweet list as follows, replace the value with the person twitter handle
tweets = extractor.user_timeline(screen_name="narendramodi", count=200)
print("Number of tweets extracted: {}.\n".format(len(tweets)))

# We print the most recent 5 tweets:
print("5 recent tweets:\n")
for tweet in tweets[:5]:
    print(tweet.text)
    print()

# We create a pandas dataframe as follows:
data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

# We display the first 10 elements of the dataframe:
display(data.head(10))
#print(dir(tweets[0]))                    this will print the available attributes of a single tweet

# uncomment below lines if you want to get the picture attached to first tweet if available , it will download the image
#print("The Media url is: {}",tweets[0].entities['media'][0]['media_url'])
#import wget
#wget.download(tweets[0].entities['media'][0]['media_url'])
#wget.download(tweets[0].entities['urls'][0]['expanded_url'])


# We add relevant data:
data['len'] = np.array([len(tweet.text) for tweet in tweets])
data['ID'] = np.array([tweet.id for tweet in tweets])
data['Date'] = np.array([tweet.created_at for tweet in tweets])
data['Source'] = np.array([tweet.source for tweet in tweets])
data['Likes'] = np.array([tweet.favorite_count for tweet in tweets])
data['RTs'] = np.array([tweet.retweet_count for tweet in tweets])

# Display of first 10 elements from dataframe:
display(data.head(10))

# We extract the mean of lenghts:
mean = np.mean(data['len'])

print("The lenght's average in tweets: {}".format(mean))

# We extract the tweet with more FAVs and more RTs:

fav_max = np.max(data['Likes'])        #calculating the tweet which got the maximum likes
rt_max = np.max(data['RTs'])           #calculating the tweet which got the maximum retweets

fav_min = np.min(data['Likes'])        #calculating the tweet which got the minimum likes
rt_min = np.min(data['RTs'])           #calculating the tweet which got the minimum retweets

fav = data[data.Likes == fav_max].index[0]
rt = data[data.RTs == rt_max].index[0]

f_min = data[data.Likes == fav_min].index[0]
r_min = data[data.RTs == rt_min].index[0];

# Max FAVs:
print("The tweet with more likes is: \n{}".format(data['Tweets'][fav]))
print("Number of likes: {}".format(fav_max))
print("{} characters.\n".format(data['len'][fav]))

# Max RTs:
print("The tweet with more retweets is: \n{}".format(data['Tweets'][rt]))
print("Number of retweets: {}".format(rt_max))
print("{} characters.\n".format(data['len'][rt]))

# Min FAVs:
print("The tweet with minimum likes is : \n{}".format(data['Tweets'][f_min]))
print("Number of likes: {}".format(data['len'][f_min]))
print("{} chracters.\n".format(data['len'][f_min]))

# Min RTs
print("The tweet with less retweets is: \n{}".format(data['Tweets'][r_min]))
print("Number of retweets: {}".format(r_min))
print("{} chracters.\n".format(data['len'][r_min]))

# We create time series for data:


tlen = pd.Series(data=data['len'].values, index=data['Date'])
tfav = pd.Series(data=data['Likes'].values, index=data['Date'])
tret = pd.Series(data=data['RTs'].values, index=data['Date'])

# Lenghts along time:, red color assigning the time for plotting the graph

tlen.plot(figsize=(16, 4), color='r')
# Likes vs retweets visualization:
tfav.plot(figsize=(16, 4), label="Likes", legend=True)
tret.plot(figsize=(16, 4), label="Retweets", legend=True);

plt.interactive(False)
plt.show(block=True)

# We obtain all possible sources:
sources = []
for source in data['Source']:
    if source not in sources:
        sources.append(source)

# We print sources list:, what was used to make tweets, example, iphone, android , linkedin , or website
print("Creation of content sources:")
for source in sources:
    print("* {}".format(source))

# We create a numpy vector mapped to labels:
percent = np.zeros(len(sources))

for source in data['Source']:
    for index in range(len(sources)):
        if source == sources[index]:
            percent[index] += 1
            pass

percent /= 100          #calculating the percentage of sources of tweets

# Now plotting the Pie chart: for showing the sources of tweets
pie_chart = pd.Series(percent, index=sources, name='Sources')
pie_chart.plot.pie(fontsize=11, autopct='%.2f', figsize=(6, 6))
plt.interactive(False)
plt.show(block=True)

#below codes are for the sentiment analysis using textblob
#first we need to remove the special character from the tweets
def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def analize_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1


# We create a column with the result of the analysis:
data['SA'] = np.array([analize_sentiment(tweet) for tweet in data['Tweets']])

# We display the updated dataframe with the new column:
display(data.head(10))
print("hello dear")

# We construct lists with classified tweets:

pos_tweets = [tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] > 0]
neu_tweets = [tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] == 0]
neg_tweets = [tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] < 0]

# We print percentages: of positive, negative, neutral tweets and shows the probability of next tweet to be
#to show the output close the graphs and pie chart
print("Percentage of positive tweets: {}%".format(len(pos_tweets) * 100 / len(data['Tweets'])))
print("Percentage of neutral tweets: {}%".format(len(neu_tweets) * 100 / len(data['Tweets'])))
print("Percentage de negative tweets: {}%".format(len(neg_tweets) * 100 / len(data['Tweets'])))

#now printing the probablity of next tweet
print("\n")
print("Probablity that the next tweet will be positive: {}".format(len(pos_tweets)/len(data['Tweets'])))
print("Probablity that the next tweet will be negative: {}".format(len(neg_tweets)/len(data['Tweets'])))
print("Probablity that the next tweet will be neutral: {}".format(len(neu_tweets)/len(data['Tweets'])))
