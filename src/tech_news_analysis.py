import zipfile
import re
import random

import pandas as pd

import nltk
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

import matplotlib.pyplot as plt
import matplotlib as mpl


MAX_FILES_READ = 1000
NUMBER_OF_CLUSTERS = 25


class Document:
    def __init__(self, filename, text):
        self.filename = filename
        self.title = text.split('\n')[0]
        self.text = text


def load_tech_news():
    tech_news = []

    with zipfile.ZipFile('./res/tech_news.zip') as z:
        count = 0
        for filename in z.namelist():
            count += 1

            if MAX_FILES_READ < count:
                break

            with z.open(filename) as f:
                text_content = f.read().decode("utf-8")
                tech_news.append(Document(filename, text_content))

    return tech_news


def tokenize(news_text):
    tokens = []

    for word in nltk.sent_tokenize(news_text):
        for token in nltk.word_tokenize(word):
            tokens.append(token.lower())

    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    return filtered_tokens


STEMMER = SnowballStemmer("english")


def steam(tokens):
    stems = []

    for token in tokens:
        stems.append(STEMMER.stem(token))

    return stems


def tokenize_and_stem(news_text):
    return steam(tokenize(news_text))


def calculate_tf_idf_matrix():

    cluster_colors = {}
    r = lambda: random.randint(0, 255)
    for c_i in range(NUMBER_OF_CLUSTERS):
        color = '#%02X%02X%02X' % (r(), r(), r())
        cluster_colors[c_i] = color

    tech_news = load_tech_news()

    tech_news_title = []
    tech_news_content = []

    for tn in tech_news:
        tech_news_title.append(tn.title)
        tech_news_content.append(tn.text)

    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in tech_news_content:
        allwords_stemmed = tokenize_and_stem(i)
        totalvocab_stemmed.extend(allwords_stemmed)

        allwords_tokenized = tokenize(i)
        totalvocab_tokenized.extend(allwords_tokenized)

    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)

    # print("There are '{}' items in our data frame.".format(str(vocab_frame.shape[0])))
    # print("Data frame contents: \n{}".format(vocab_frame.head()))

    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.8,
        max_features=200000,
        min_df=0.2,
        stop_words='english',
        use_idf=True,
        tokenizer=tokenize_and_stem,
        ngram_range=(1, 3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(tech_news_content)

    # print("TF-IDF matrix shape: {}".format(tfidf_matrix.shape))
    terms = tfidf_vectorizer.get_feature_names()
    dist = 1 - cosine_similarity(tfidf_matrix)
    # print("Dist: {}".format(dist))

    km = KMeans(n_clusters=NUMBER_OF_CLUSTERS)
    km.fit(tfidf_matrix)

    clusters = km.labels_.tolist()
    print("Clusters: {}".format(clusters))

    tech_news_data = {"title": tech_news_title, "cluster": clusters}
    frame = pd.DataFrame(tech_news_data, index=[clusters], columns=["title", "cluster"])
    print("Number of movies per cluster: \n{}".format(frame["cluster"].value_counts()))

    print("Top terms per cluster:\n")

    # Sort cluster centers by proximity to centroid.
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    # Helper function
    def get_cluster_words(cluster, n=6):
        words = []
        for ind in order_centroids[cluster, :n]:  # Print 6 words per cluster
            words.append(vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0])
        return ", ".join(words)

    for i in range(NUMBER_OF_CLUSTERS):
        print("Cluster {} words: {}".format(i, get_cluster_words(i)))

        print("Cluster {} titles:".format(i), end='')
        for title in frame.loc[i]['title'].values.tolist():
            print(" {},".format(title), end='')
        print("\n")

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

    # Shape of the result will be (n_components, n_samples).
    pos = mds.fit_transform(dist)
    xs, ys = pos[:, 0], pos[:, 1]

    cluster_names = dict([(i, get_cluster_words(i, 5)) for i in range(NUMBER_OF_CLUSTERS)])

    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=tech_news_title))

    # Group by cluster.
    groups = df.groupby('label')

    # Set up plot.
    fig, ax = plt.subplots(figsize=(17, 9))  # set size
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    # Iterate through groups to layer the plot.
    # Note that we use the cluster_name and cluster_color dicts with the 'name'
    # lookup to return the appropriate color/label.
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                label=cluster_names[name], color=cluster_colors[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(
            axis='y',  # changes apply to the y-axis
            which='both',  # both major and minor ticks are affected
            left='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelleft='off')

    ax.legend(numpoints=1)  # show legend with only 1 point

    # Add label in x,y position with the label as the film title.
    for i in range(len(df)):
        ax.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['title'], size=8)

        # Uncomment the below to show or save the plot.
    plt.show()  # show the plot
    # plt.savefig('clusters_small_noaxes.png', dpi=200) # save the plot as an image

    plt.close()


if __name__ == "__main__":
    calculate_tf_idf_matrix()


"""

from scipy.cluster.hierarchy import ward, dendrogram

# Define the linkage_matrix using ward clustering pre-computed distances.
linkage_matrix = ward(dist) 

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=titles);

plt.tick_params(\
    axis= 'x',         # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

# Uncomment the below to show or save the plot.
plt.show()
#plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters
plt.close()

"""