import requests
import pandas as pd
import nltk
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure VADER lexicon is available
nltk.download('vader_lexicon')

# Set your News API key and construct the query URL for articles containing "technology"
api_key = '9958ca2f9c2c414584a244097942b162'
url = f'https://newsapi.org/v2/everything?q=technology&language=en&apiKey={api_key}'

# Fetch data from the News API
response = requests.get(url)
if response.status_code != 200:
    raise Exception(f"Error fetching data: {response.status_code} - {response.text}")
data = response.json()

# Extract articles (if any)
articles = data.get('articles', [])

# Print Raw Data in a Dictionary-Like Structure 
articles_dict = {}
for i, article in enumerate(articles):
    articles_dict[f"news-{i}"] = {
        "urlToImage": article.get("urlToImage"),
        "publishedAt": article.get("publishedAt"),
        "author": article.get("author"),
        "title": article.get("title"),
        "description": article.get("description"),
        "url": article.get("url")
    }
print(articles_dict)

# Create a DataFrame with the raw data
df_raw = pd.DataFrame(articles)
print("\nRaw DataFrame from News API:")
print(df_raw.head())

# --- Text Cleaning and Sentiment Analysis ---

# Function to clean text: lowercasing, removing numbers, extra spaces, and punctuation
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)         # remove numbers
    text = re.sub(r'\s+', ' ', text)         # collapse multiple spaces
    text = re.sub(r'[^\w\s]', '', text)      # remove punctuation
    return text.strip()

# Combine title and description for a fuller text (if available)
def combine_text(row):
    title = row.get('title', '')
    description = row.get('description', '')
    return f"{title} {description}"

# Create a combined text column and then a cleaned version
df_raw['combined_text'] = df_raw.apply(combine_text, axis=1)
df_raw['clean_text'] = df_raw['combined_text'].apply(clean_text)

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Updated function to label sentiment with five labels based on compound score
def get_sentiment_label(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.5:
        return 'very positive'
    elif score > 0.05:
        return 'positive'
    elif score < -0.5:
        return 'very negative'
    elif score < -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis to each cleaned text entry
df_raw['sentiment'] = df_raw['clean_text'].apply(get_sentiment_label)

print(df_raw[['title', 'clean_text', 'sentiment']].head())

#  Vectorization

# Vectorize the cleaned text using TF-IDF (ignoring common English stop words)
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df_raw['clean_text'])

# Convert the TF-IDF matrix to a DataFrame
df_vectorized = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Convert TF-IDF scores to whole numbers by multiplying by 100 and rounding
df_vectorized = df_vectorized.multiply(100).round(0).astype(int)

# Append the sentiment labels for each article
df_vectorized['sentiment'] = df_raw['sentiment'].values

print(df_vectorized.head(100))

df_vectorized.to_csv('vectorized_data_newsapi.csv', index=False)

