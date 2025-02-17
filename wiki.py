import requests
import pandas as pd
import nltk
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure VADER lexicon is available
nltk.download('vader_lexicon')

# Your client ID (for tracking purposes, optional)
client_id = 'f44224c05c7583342316b644e0c68ea0'

# Set up headers including the client ID if desired
headers = {
    "X-Client-ID": client_id  # Optional custom header
}

# Set query parameters to search for "tech" in Wikipedia and increase limit to 20 results
url = "https://en.wikipedia.org/w/api.php"
params = {
    "action": "query",
    "list": "search",
    "srsearch": "tech",
    "srlimit": 100,  # Request 20 results instead of the default 10
    "format": "json"
}

# Fetch data from the Wikipedia API (no authentication needed for basic search)
response = requests.get(url, params=params, headers=headers)
if response.status_code != 200:
    raise Exception(f"Error fetching data: {response.status_code} - {response.text}")
data = response.json()

# Extract search results
articles = data.get("query", {}).get("search", [])

#Print Raw Data
articles_dict = {}
for i, article in enumerate(articles):
    # Remove HTML tags from the snippet if it exists
    snippet_text = article.get("snippet", "")
    snippet_clean = re.sub(r'<.*?>', '', snippet_text) if snippet_text else ""
    articles_dict[f"wiki-{i}"] = {
        "title": article.get("title", ""),
        "snippet": snippet_clean,
        "pageid": article.get("pageid")
    }
print(articles_dict)

# Create a DataFrame with the raw data
df_raw = pd.DataFrame(articles)

# Ensure 'snippet' exists in the DataFrame; if not, create it as an empty column
if 'snippet' not in df_raw.columns:
    df_raw['snippet'] = ""

# Clean the snippet column to remove HTML tags
df_raw['snippet'] = df_raw['snippet'].apply(lambda x: re.sub(r'<.*?>', '', x) if isinstance(x, str) else "")
print("\nRaw DataFrame from Wikipedia API:")
print(df_raw.head())

# Text Cleaning and Sentiment Analysis#

# Function to clean text: lowercasing, removing numbers, extra spaces, and punctuation
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)         # remove numbers
    text = re.sub(r'\s+', ' ', text)         # collapse multiple spaces
    text = re.sub(r'[^\w\s]', '', text)      # remove punctuation
    return text.strip()

# Combine title and snippet for a fuller text representation
def combine_text(row):
    title = row.get("title", "")
    snippet = row.get("snippet", "")
    return f"{title} {snippet}"

# Create a combined text column and then a cleaned version
df_raw['combined_text'] = df_raw.apply(combine_text, axis=1)
df_raw['clean_text'] = df_raw['combined_text'].apply(clean_text)

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Updated function to label sentiment with five labels based on the compound score
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

# --- Text Vectorization ---

# Vectorize the cleaned text using TF-IDF (ignoring common English stop words)
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df_raw['clean_text'])
df_vectorized = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Convert TF-IDF scores to whole numbers by multiplying by 100 and rounding
df_vectorized = df_vectorized.multiply(100).round(0).astype(int)

# Append the sentiment labels for each article
df_vectorized['sentiment'] = df_raw['sentiment'].values

print(df_vectorized.head(100))

# Export the DataFrame to a CSV file
df_vectorized.to_csv('vectorized_data_wiki.csv', index=False)
