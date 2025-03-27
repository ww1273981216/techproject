## What to import
import requests  ## for getting data from a server
import re   ## for regular expressions
import pandas as pd    ## for dataframes and related
from pandas import DataFrame

## To tokenize and vectorize text type data
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
## For word clouds
## conda install -c conda-forge wordcloud
## May also have to run conda update --all on cmd
#import PIL
#import Pillow
#import wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
import random as rd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
#from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree

from sklearn.decomposition import LatentDirichletAllocation 
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import silhouette_samples, silhouette_score
import sklearn
from sklearn.cluster import KMeans

from sklearn import preprocessing

import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import ward, dendrogram


# Load the vectorized data
path = '/Users/willweiwu/Library/CloudStorage/Dropbox/CU Boulder/2025 Spring/INFO 5653-001 Text Mining/Project/wiki/vectorized_data_wiki.csv'
df = pd.read_csv(path)

# Drop the sentiment column
df = df.drop(columns=['sentiment'])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df.columns)

NUM_TOPICS = 5
lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=100000, learning_method='online')
lda_Z_DF = lda_model.fit_transform(df)
print(lda_Z_DF.shape)

def print_topics(model, feature_names, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(feature_names[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]])

# Print the topics
print("LDA Model:")
print_topics(lda_model, df.columns)

plt.figure(figsize=(50,30))
word_topic = np.array(lda_model.components_).transpose()
num_top_words = 15
vocab_array = np.asarray(df.columns)
fontsize_base = 40

for t in range(NUM_TOPICS):
    plt.subplot(1, NUM_TOPICS, t + 1)
    plt.ylim(0, num_top_words + 0.5)
    plt.xticks([])
    plt.yticks([])
    plt.title('Topic #{}'.format(t))
    top_words_idx = np.argsort(word_topic[:, t])[::-1][:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words - i - 0.5, word, fontsize=fontsize_base / 2)

plt.savefig("TopicsVis.pdf")
plt.show()

import pyLDAvis
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Load the vectorized data
path = '/Users/willweiwu/Library/CloudStorage/Dropbox/CU Boulder/2025 Spring/INFO 5653-001 Text Mining/Project/wiki/vectorized_data_wiki.csv'
df = pd.read_csv(path)

# Drop the sentiment column if it exists
if 'sentiment' in df.columns:
    df = df.drop(columns=['sentiment'])

# Convert the data into a matrix
X = df.values

# Fit the LDA model
NUM_TOPICS = 5
lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10000, learning_method='online')
lda_Z_DF = lda_model.fit_transform(X)

# Prepare the inputs for pyLDAvis
vocab = df.columns
term_frequency = np.asarray(X.sum(axis=0)).flatten()
doc_lengths = np.sum(X, axis=1).tolist()

# Visualize the topics using pyLDAvis
pyLDAvis.enable_notebook()
vis_data = pyLDAvis.prepare(
    topic_term_dists=lda_model.components_,
    doc_topic_dists=lda_Z_DF,
    doc_lengths=doc_lengths,
    vocab=vocab,
    term_frequency=term_frequency
)

# Save the visualization
pyLDAvis.save_html(vis_data, 'lda_visualization.html')
print("Visualization saved as 'lda_visualization.html'")



