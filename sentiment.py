pip install tweepy textblob matplotlib pandas nltk

import tweepy,re
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

pip install tweepy --upgrade

# authentication
bearerToken='AAAAAAAAAAAAAAAAAAAAAPW7pwEAAAAA4vnqWZc8Isfu3qJWWzbWNllcun4%3DLV7bel4tM13Oq80J5amxHvLbkwiwaa04IgU3SH7n4s8wfT11Rh',
consumerKey = '9RGl4MovrlDHHGHhok0xWdxZi'
consumerSecret = 'C1Lo0XyWOgnY8HhMsvAXezOH6VrKGBHvU4hsYgi4EquJwN3cFw'
accessToken = '1003281934154285056-V72ClznwqwkjdXp06cqHbGuYpyeq8H'
accessTokenSecret = 'BUvQXJbTert094Boaiteu9nR6i8W2nhvEaew9eI6SZVS1'
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)

import tweepy
import pandas as pd

# Authentication with Client
client = tweepy.Client(
    bearer_token='AAAAAAAAAAAAAAAAAAAAAPW7pwEAAAAA4vnqWZc8Isfu3qJWWzbWNllcun4%3DLV7bel4tM13Oq80J5amxHvLbkwiwaa04IgU3SH7n4s8wfT11Rh',
    consumer_key='9RGl4MovrlDHHGHhok0xWdxZi',
    consumer_secret='C1Lo0XyWOgnY8HhMsvAXezOH6VrKGBHvU4hsYgi4EquJwN3cFw',
    access_token='1003281934154285056-V72ClznwqwkjdXp06cqHbGuYpyeq8H',
    access_token_secret='BUvQXJbTert094Boaiteu9nR6i8W2nhvEaew9eI6SZVS1'
)

# User Input
searchTerm = input("Enter Keyword/Tag to search about: ")
NoOfTerms = int(input("Enter how many tweets to search: "))

# Fetch Tweets using v2 API
try:
    response = client.search_recent_tweets(
        query=searchTerm + " -is:retweet",
        max_results=min(NoOfTerms, 100),  # Max 100 per request in v2
        tweet_fields=['created_at', 'public_metrics'],
        user_fields=['username']
    )

    # Process tweets
    if response.data:
        tweet_list = [
            {
                "Tweet_Text": tweet.text,
                "Created_At": tweet.created_at,
                "Likes": tweet.public_metrics['like_count'],
                "Retweets": tweet.public_metrics['retweet_count']
            }
            for tweet in response.data
        ]
        tweet_df = pd.DataFrame(tweet_list)
        print(tweet_df.head())
    else:
        print("No tweets found for this query.")

except tweepy.TweepyException as e:
    print("Error:", e)
except Exception as e:
    print("General Error:", e)


import pandas as pd

# Assuming tweet_list already contains the tweet data
# Example tweet_list:
# tweet_list = [
#     {"Tweet_Text": "Sample tweet 1", "Created_At": "2024-06-20", "Likes": 10, "Retweets": 2},
#     {"Tweet_Text": "Sample tweet 2", "Created_At": "2024-06-21", "Likes": 15, "Retweets": 3}
# ]

# Convert to DataFrame
tweet_df = pd.DataFrame(tweet_list)

# Display the DataFrame
print(tweet_df)

# For Jupyter Notebooks specifically, display it in a table format:
from IPython.display import display
display(tweet_df)



import re

# Function to clean text
def clean_data(text):
    return ' '.join(re.sub(r"(@[a-zA-Z0-9]+)|([^0-9A-Za-z\s])|(https?://\S+)", " ", text).split())

# Apply cleaning function to the correct column
tweet_df['cleaned_data'] = tweet_df['Tweet_Text'].apply(clean_data)

# Display updated DataFrame
from IPython.display import display
display(tweet_df)


# Function to drop numbers from a single string
def drop_numbers(text):
    return ''.join(char for char in text if not char.isdigit())

# Apply the function to the 'cleaned_data' column
tweet_df['cleaned_data'] = tweet_df['cleaned_data'].apply(drop_numbers)

# Display updated DataFrame
from IPython.display import display
display(tweet_df)

# Lowercase all text directly
tweet_df['cleaned_data'] = tweet_df['cleaned_data'].str.lower()

# Display the updated DataFrame
from IPython.display import display
display(tweet_df)

import nltk

nltk.download('punkt')  # Sentence tokenizer
nltk.download('wordnet')  # WordNet for lemmatization
nltk.download('omw-1.4')  # Open Multilingual WordNet
nltk.download('averaged_perceptron_tagger')  # POS tagging



import nltk

# Force download the punkt resource again
nltk.download('punkt_tab', force=True)


import nltk.data
print(nltk.data.path)


nltk.data.path.append('/Users/ashwinikumar/nltk_data')

from nltk.tokenize import word_tokenize

# Test tokenization
text = "This is a test sentence."
tokens = word_tokenize(text)
print(tokens)


import nltk
nltk.download('averaged_perceptron_tagger_eng')

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

# Enhanced Lemmatization Function
def lemmatise_with_pos(text):
    try:
        text_tokens = word_tokenize(text)  # Tokenize the text
        pos_tags = pos_tag(text_tokens)  # POS tagging
        text_lemm = [
            lemmatizer.lemmatize(word, get_wordnet_pos(tag)) 
            for word, tag in pos_tags
        ]
        return ' '.join(text_lemm)
    except Exception as e:
        print(f"Error processing text: {text}, Error: {e}")
        return text  # Return original text in case of error

# Apply to the 'cleaned_data' column
tweet_df['cleaned_data'] = tweet_df['cleaned_data'].apply(lemmatise_with_pos)

# Display updated DataFrame
from IPython.display import display
display(tweet_df)

tweet_df['cleaned_data'].values

nltk.download('stopwords')

def remove_stopword(text):
    text_tokens = word_tokenize(text)
    tokens = [word for word in text_tokens if not word in set(stopwords.words('english'))]
    tokens_text = ' '.join(tokens)
    return tokens_text

tweet_df['cleaned_data'] = tweet_df['cleaned_data'].apply(remove_stopword)

tweet_df['cleaned_data'].values

# Lets calculate the Polarity of the Reviews
def get_polarity(text):
    textblob = TextBlob(str(text))
    pol = textblob.sentiment.polarity
    if(pol==0):
        return "Neutral"
    elif(pol>0 and pol<=0.3):
        return "Weakly Positive"
    elif(pol>0.3 and pol<=0.6):
        return "Positive"
    elif(pol>0.6 and pol<=1):
        return "Strongly Positive"
    elif(pol>-0.3 and pol<=0):
        return "Weakly Negative"
    elif(pol>-0.6 and pol<=-0.3):
        return "Negative"
    elif(pol>-1 and pol<=-0.6):
        return "Strongly Negative"
    
tweet_df['polarity'] = tweet_df['cleaned_data'].apply(get_polarity)

tweet_df['polarity'].value_counts()

tweet_df.dtypes

neutral = 0
wpositive = 0
spositive = 0
positive = 0
negative = 0
wnegative = 0
snegative = 0
polarity = 0

for i in range(0, NoOfTerms):
    textblob = TextBlob(str(tweet_df['cleaned_data'][i]))
    polarity+= textblob.sentiment.polarity
    pol = textblob.sentiment.polarity
    if (pol == 0):  # adding reaction of how people are reacting to find average later
        neutral += 1
    elif (pol > 0 and pol <= 0.3):
        wpositive += 1
    elif (pol > 0.3 and pol <= 0.6):
        positive += 1
    elif (pol > 0.6 and pol <= 1):
        spositive += 1
    elif (pol > -0.3 and pol <= 0):
        wnegative += 1
    elif (pol > -0.6 and pol <= -0.3):
        negative += 1
    elif (pol > -1 and pol <= -0.6):
        snegative += 1


# finding average reaction
polarity = polarity / NoOfTerms
polarity

def percentage(part, whole):
    temp = 100 * float(part) / float(whole)
    return format(temp, '.2f')

 # finding average of how people are reacting
positive = percentage(positive, NoOfTerms)
wpositive = percentage(wpositive, NoOfTerms)
spositive = percentage(spositive, NoOfTerms)
negative = percentage(negative, NoOfTerms)
wnegative = percentage(wnegative, NoOfTerms)
snegative = percentage(snegative, NoOfTerms)
neutral = percentage(neutral, NoOfTerms)



 # printing out data
print("How people are reacting on " + searchTerm + " by analyzing " + str(NoOfTerms) + " tweets.")
print()
print("-----------------------------------------------------------------------------------------")
print()
print("General Report: ")

if (polarity == 0):
    print("Neutral")
elif (polarity > 0 and polarity <= 0.3):
    print("Weakly Positive")
elif (polarity > 0.3 and polarity <= 0.6):
    print("Positive")
elif (polarity > 0.6 and polarity <= 1):
    print("Strongly Positive")
elif (polarity > -0.3 and polarity <= 0):
    print("Weakly Negative")
elif (polarity > -0.6 and polarity <= -0.3):
    print("Negative")
elif (polarity > -1 and polarity <= -0.6):
    print("Strongly Negative")

print()
print("------------------------------------------------------------------------------------------")
print()
print("Detailed Report: ")
print(str(positive) + "% people thought it was positive")
print(str(wpositive) + "% people thought it was weakly positive")
print(str(spositive) + "% people thought it was strongly positive")
print(str(negative) + "% people thought it was negative")
print(str(wnegative) + "% people thought it was weakly negative")
print(str(snegative) + "% people thought it was strongly negative")
print(str(neutral) + "% people thought it was neutral")

import matplotlib.pyplot as plt

# Data Preparation
sizes = [positive, wpositive, spositive, neutral, negative, wnegative, snegative]
colors = ['#2ecc71', '#58d68d', '#1e8449', '#f4d03f', '#e74c3c', '#ec7063', '#c0392b']
labels = [
    f'Positive [{positive}%]', 
    f'Weakly Positive [{wpositive}%]',
    f'Strongly Positive [{spositive}%]', 
    f'Neutral [{neutral}%]',
    f'Negative [{negative}%]', 
    f'Weakly Negative [{wnegative}%]', 
    f'Strongly Negative [{snegative}%]'
]

# Exploding slices for emphasis
explode = (0.05, 0.05, 0.1, 0, 0.05, 0.05, 0.1)  # Highlight Strongly Positive and Strongly Negative

# Plotting the Pie Chart
fig, ax = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    explode=explode,
    autopct='%1.1f%%',  # Show percentage with one decimal
    startangle=140,
    textprops={'fontsize': 12}
)

# Styling Autotexts
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(10)

# Title and Legend
plt.title(f"Sentiment Analysis on '{searchTerm}' ({NoOfTerms} Tweets)", fontsize=14, weight='bold')
plt.legend(wedges, labels, title="Sentiment Types", loc="center left", bbox_to_anchor=(1, 0.5, 0, 1))

# Ensures Pie is Circular
plt.axis('equal')

# Display Chart
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Example sentiment data
# Ensure 'Sentiment' column exists in tweet_df
sentiment_counts = tweet_df['polarity'].value_counts(normalize=True) * 100  # Convert to percentages

# Plotting the Bar Graph
plt.figure(figsize=(12, 8))
bar_plot = sns.barplot(
    x=sentiment_counts.index, 
    y=sentiment_counts.values, 
)

# Add percentage labels on top of each bar
for bar in bar_plot.patches:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f'{bar.get_height():.1f}%',  # Format as percentage
        ha='center',
        va='bottom',
        fontsize=10,
        color='black'
    )

# Customize Graph
plt.title(f"Sentiment Distribution for '{searchTerm}' ({NoOfTerms} by analyzing Tweets)", fontsize=14, weight='bold')
plt.xlabel('polarity', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Display the Graph
plt.tight_layout()
plt.show()


