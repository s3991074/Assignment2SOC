@ -1,515 +0,0 @@
# Code for Assignment 1
# Author is Gaurav Shiven Sursen S3941388


import sys
from googleapiclient.discovery import build
import json
import string
from collections import Counter
import nltk
import datetime
import matplotlib.pyplot as plt
import re
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud


nltk.download('vader_lexicon')

nltk.download('stopwords')


# ============================================================
# YouTube Client
# ============================================================

def youtubeClient():
    try:
        apiKey = "API_KEY"
        youtube = build('youtube', 'v3', developerKey=apiKey)
    except Exception as e:
        sys.stderr.write(f"Failed to create YouTube client: {e}\n")
        sys.exit(1)
    return youtube


# ============================================================
# Fetch YouTube Data
# ============================================================

def fetchYoutubeData(searchQuery, maxVideos=25, maxCommentsPerVideo=250, outputFile='PokemonNuzlockeData.json'):
    client = youtubeClient()
    print("YouTube client created successfully.\n")

    currenttime = datetime.datetime.now(datetime.timezone.utc)
    twoyearsago = currenttime - datetime.timedelta(days=365 * 2)

    searchResponse = client.search().list(
        q=searchQuery,
        part='snippet',
        type='video',
        relevanceLanguage='en',
        publishedAfter=twoyearsago.isoformat(),
        order='viewCount',
        maxResults=min(maxVideos, 50)
    ).execute()

    videoIds = []
    videoSnippets = {}
    for item in searchResponse.get('items', []):
        videoId = item['id']['videoId']
        videoIds.append(videoId)
        videoSnippets[videoId] = item['snippet']

    print(f"Found {len(videoIds)} videos.\n")

    statsResponse = client.videos().list(
        id=','.join(videoIds),
        part='statistics'
    ).execute()

    videoStats = {item['id']: item['statistics'] for item in statsResponse.get('items', [])}

    videos = []
    for videoId in videoIds:
        snippet = videoSnippets[videoId]
        stats = videoStats.get(videoId, {})

        video = {
            'title': snippet['title'],
            'videoId': videoId,
            'channelTitle': snippet['channelTitle'],
            'publishedAt': snippet['publishedAt'],
            'viewCount': int(stats.get('viewCount', 0)),
            'likeCount': int(stats.get('likeCount', 0)),
            'comments': []
        }

        try:
            commentResponse = client.commentThreads().list(
                videoId=videoId,
                part='snippet',
                maxResults=maxCommentsPerVideo,
                textFormat='plainText'
            ).execute()

            for commentThread in commentResponse.get('items', []):
                topComment = commentThread['snippet']['topLevelComment']['snippet']
                video['comments'].append({
                    'author': topComment['authorDisplayName'],
                    'text': topComment['textDisplay'],
                    'publishedAt': topComment['publishedAt'],
                    'likeCount': topComment.get('likeCount', 0)
                })

            print(f"{snippet['title'][:50]}... → {len(video['comments'])} comments")
        except Exception as e:
            print(f"{snippet['title'][:50]}... → Comments disabled or error: {e}")

        videos.append(video)

    with open(outputFile, 'w', encoding='utf-8') as f:
        json.dump({'videos': videos}, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Saved {len(videos)} videos to '{outputFile}'.")


# ============================================================
# Fetch Top Uploaders
# ============================================================

def getTopUploaders(searchQuery, maxResults=50, outputFile='PokemonNuzlockeTopUploaders.json'):
    client = youtubeClient()
    lPopularChannels, lRelevantChannels = [], []

    currenttime = datetime.datetime.now(datetime.timezone.utc)
    twoyearsago = currenttime - datetime.timedelta(days=365 * 2)

    # Popular videos
    searchResponse = client.search().list(
        q=searchQuery,
        part='snippet',
        type='video',
        relevanceLanguage='en',
        publishedAfter=twoyearsago.isoformat(),
        order='viewCount',
        maxResults=maxResults
    ).execute()

    for item in searchResponse.get('items', []):
        lPopularChannels.append(item['snippet']['channelTitle'])

    # Relevant videos
    searchResponse = client.search().list(
        q=searchQuery,
        part='snippet',
        type='video',
        relevanceLanguage='en',
        order='relevance',
        maxResults=maxResults
    ).execute()

    for item in searchResponse.get('items', []):
        lRelevantChannels.append(item['snippet']['channelTitle'])

    lPopularChannels.extend(lRelevantChannels)
    hChannelCount = Counter(lPopularChannels)

    print("\nTop uploaders:\n")
    for channel, count in hChannelCount.most_common(20):
        print(f"{channel}: {count}")

    with open(outputFile, 'w', encoding='utf-8') as f:
        json.dump({"channels": dict(hChannelCount.most_common())}, f, indent=2)

    print(f"\nSaved top uploader data to '{outputFile}'.")


# ============================================================
# Text Processor Class
# ============================================================

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F900-\U0001F9FF"
        "\U00002700-\U000027BF"
        "\U00002600-\U000026FF"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)

class TextProcessor:
    def __init__(self, tokenizer, stopwords):
        self.tokenizer = tokenizer
        self.stopwords = stopwords

    def process(self, text):
        text = remove_emojis(text.lower())
        tokens = self.tokenizer.tokenize(text)
        tokensStripped = [tok.strip() for tok in tokens]
        regexDigit = re.compile(r"\d+")
        regexHttp = re.compile(r"^http")
        return [tok for tok in tokensStripped
                if tok not in self.stopwords
                and regexDigit.match(tok) is None
                and regexHttp.match(tok) is None]


# ============================================================
# Sentiment Functions
# ============================================================
def computeSentiment(tokens, posWords, negWords):
    pos = len([t for t in tokens if t in posWords])
    neg = len([t for t in tokens if t in negWords])
    return pos - neg

def classify_sentiment(score):
    if score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else:
        return 'neutral'
        
# ============================================================
# Main Execution
# ============================================================

if __name__ == '__main__':
    SEARCH_QUERY = 'Pokemon Nuzlocke'
    MAX_VIDEOS = 50
    MAX_COMMENTS = 100
    VIDEO_OUTPUT_FILE = 'PokemonNuzlockeData.json'
    UPLOADER_OUTPUT_FILE = 'PokemonNuzlockeTopUploaders.json'

    # Fetch videos + comments
    fetchYoutubeData(SEARCH_QUERY, MAX_VIDEOS, MAX_COMMENTS, VIDEO_OUTPUT_FILE)

    # Fetch top uploaders
    getTopUploaders(SEARCH_QUERY, MAX_VIDEOS, UPLOADER_OUTPUT_FILE)

    # Setup NLP tools
    # custom stopwords are used after the initial graph
    custom_stopwords = [
    'pokemon', 'part', '...', '#pokemon'
]
    tokenizer = nltk.tokenize.TweetTokenizer()
    punct = list(string.punctuation)
    stopwords = nltk.corpus.stopwords.words('english') + punct + ['via'] + custom_stopwords
  
    processor = TextProcessor(tokenizer, stopwords)
    stemmer = nltk.stem.PorterStemmer()

    # ---------------------------
    # Analyze Video Titles
    # ---------------------------
    termFreqCounter = Counter()
    try:
        with open(VIDEO_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            videosData = json.load(f)
        for video in videosData['videos']:
            title = re.sub(u"(\u2018|\u2019|\u2014)", "", video.get('title', ''))
            tokens = processor.process(title)
            tokens = [stemmer.stem(tok) for tok in tokens]
            termFreqCounter.update(tokens)
    except FileNotFoundError:
        print(f"{VIDEO_OUTPUT_FILE} not found. Run fetchYoutubeData first.")

    print("\nTop 50 terms from video titles:\n")
    for term, count in termFreqCounter.most_common(50):
        print(f"{term}: {count}")

    y = [count for _, count in termFreqCounter.most_common(50)]
    x = range(1, len(y) + 1)
    plt.figure(figsize=(10, 10))
    plt.bar(x, y)
    plt.xticks(x, [term for term, _ in termFreqCounter.most_common(50)], rotation=90)
    plt.title("Term frequency distribution (YouTube video titles)")
    plt.ylabel("# of words with term frequency")
    plt.xlabel("Term frequency")
    plt.show()

    # ---------------------------
    # Analyze Top Uploaders
    # ---------------------------
    uploaderFreqCounter = Counter()
    try:
        with open(UPLOADER_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            uploaderData = json.load(f)
        uploaderFreqCounter.update(uploaderData['channels'])
    except FileNotFoundError:
        print(f"{UPLOADER_OUTPUT_FILE} not found. Run getTopUploaders first.")

    print("\nTop 20 uploaders:\n")
    for channel, count in uploaderFreqCounter.most_common(20):
        print(f"{channel}: {count}")

    y = [count for _, count in uploaderFreqCounter.most_common(20)]
    x = range(1, len(y) + 1)
    plt.figure(figsize=(10, 6))
    plt.bar(x, y)
    plt.xticks(x, [channel for channel, _ in uploaderFreqCounter.most_common(20)], rotation=90)
    plt.title("Top 20 YouTube Uploaders")
    plt.ylabel("# of videos in top results")
    plt.xlabel("Channel")
    plt.show()

# ---------------------------
# Sentiment Analysis + Comparison
# ---------------------------

# ---- word lists ----
setPosWords = {
    'good', 'great', 'awesome', 'amazing', 'love', 'best',
    'fun', 'nice', 'excellent', 'fantastic', 'cool'
}

setNegWords = {
    'bad', 'worst', 'hate', 'boring', 'terrible',
    'awful', 'poor', 'sad', 'annoying', 'difficult'
}

commentSentiments = []

# ---- Opinion sentiment ----
for video in videosData['videos']:
    for comment in video['comments']:
        text = comment.get('text', '')
        tokens = processor.process(text) #this cleans the text
        tokens = [stemmer.stem(tok) for tok in tokens]

        score = computeSentiment(tokens, setPosWords, setNegWords) #gives a score to each comment depending on amount of pos/neg words
        commentSentiments.append(score) #creates list of comments

# ---- Count opinion labels ----
opinion_labels = [classify_sentiment(s) for s in commentSentiments] #turns numbers into pos/neg/neut
opinion_counts = Counter(opinion_labels) #counts results

# ---- VADER sentiment ----
sia = SentimentIntensityAnalyzer() #starts VADER method
vader_labels = []

for video in videosData['videos']:
    for comment in video['comments']:
        text = comment.get('text', '')
        score = sia.polarity_scores(text)['compound'] #returns the value between -1 and +1

        if score >= 0.05:
            vader_labels.append('positive')
        elif score <= -0.05:
            vader_labels.append('negative')
        else:
            vader_labels.append('neutral') #classify data using thresholds

vader_counts = Counter(vader_labels) #count results


# ---- Create table ----
df = pd.DataFrame({
    'Basic Analysis': [
        opinion_counts.get('positive', 0),
        opinion_counts.get('neutral', 0),
        opinion_counts.get('negative', 0)
    ],
    'Vader Analysis': [
        vader_counts.get('positive', 0),
        vader_counts.get('neutral', 0),
        vader_counts.get('negative', 0)
    ]
}, index=['positive count', 'neutral count', 'negative count'])

print("\nSentiment Comparison:\n")
print(df)


# ---------------------------
# Word Cloud For Vader Method
# ---------------------------
vader_labels = []

vader_positive_text = []
vader_negative_text = []
vader_neutral_text = []

for video in videosData['videos']:
    for comment in video['comments']:
        text = comment.get('text', '')

        score = sia.polarity_scores(text)['compound']

        tokens = processor.process(text)
        tokens = [stemmer.stem(tok) for tok in tokens]

        if score >= 0.05:
            vader_labels.append('positive')
            vader_positive_text.extend(tokens)

        elif score <= -0.05:
            vader_labels.append('negative')
            vader_negative_text.extend(tokens)

        else:
            vader_labels.append('neutral')
            vader_neutral_text.extend(tokens)

# --------------- positive word cloud ---------------
vader_pos_text = " ".join(vader_positive_text)

vader_pos_wc = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='Greens'
).generate(vader_pos_text)

plt.figure(figsize=(10,5))
plt.imshow(vader_pos_wc, interpolation='bilinear')
plt.axis('off')
plt.title("VADER Positive Word Cloud")
plt.show()

# --------------- negative word cloud ---------------
vader_neg_text = " ".join(vader_negative_text)

vader_neg_wc = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='Reds'
).generate(vader_neg_text)

plt.figure(figsize=(10,5))
plt.imshow(vader_neg_wc, interpolation='bilinear')
plt.axis('off')
plt.title("VADER Negative Word Cloud")
plt.show()

# --------------- neutral word cloud ---------------
vader_neutral_wc = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='Blues'
).generate(" ".join(vader_neutral_text))

plt.figure(figsize=(10,5))
plt.imshow(vader_neutral_wc, interpolation='bilinear')
plt.axis('off')
plt.title("VADER Neutral Word Cloud")
plt.show()


# ---------------------------
# Word Cloud For Opinion Method
# ---------------------------
positive_text = []
negative_text = []
neutral_text = []

for video in videosData['videos']:
    for comment in video['comments']:
        text = comment.get('text', '')
        tokens = processor.process(text)
        tokens = [stemmer.stem(tok) for tok in tokens]

        score = computeSentiment(tokens, setPosWords, setNegWords)

        if score > 0:
            positive_text.extend(tokens)
        elif score < 0:
            negative_text.extend(tokens)
        else:
            neutral_text.extend(tokens)

# --------------- positive word cloud ---------------
pos_text = " ".join(positive_text)

pos_wc = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='Greens'
).generate(pos_text)

plt.figure(figsize=(10,5))
plt.imshow(pos_wc, interpolation='bilinear')
plt.axis('off')
plt.title("Positive Sentiment Word Cloud")
plt.show()

# --------------- negative word cloud ---------------
neg_text = " ".join(negative_text)

neg_wc = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='Reds'
).generate(neg_text)

plt.figure(figsize=(10,5))
plt.imshow(neg_wc, interpolation='bilinear')
plt.axis('off')
plt.title("Negative Sentiment Word Cloud")
plt.show()

# --------------- neutral word cloud ---------------
neutral_wc = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='Blues'
).generate(" ".join(neutral_text))

plt.figure(figsize=(10,5))
plt.imshow(neutral_wc, interpolation='bilinear')
plt.axis('off')
plt.title("Neutral Sentiment Word Cloud")
plt.show()