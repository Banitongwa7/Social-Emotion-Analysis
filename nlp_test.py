from nltk.corpus import twitter_samples

# Get the current working directory
current_path = os.getcwd()

positive_tweets = twitter_samples.strings('data/twitter/positive_tweets.json')
negative_tweets = twitter_samples.strings('data/twitter/negative_tweets.json')
text = twitter_samples.strings('data/twitter/tweets.20150430-223406.json')