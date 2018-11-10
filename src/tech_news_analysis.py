from typing import List

import zipfile
import re
import random as rnd

import pandas as pd

import nltk
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

import matplotlib.pyplot as plt
import matplotlib as mpl


MAX_FILES_READ = 100
NUMBER_OF_CLUSTERS = 10


class Document:
    def __init__(self, filename, text):
        self.filename = filename
        self.title = text.split('\n')[0]
        self.text = text


class CorpusMetadata:
    def __init__(self, documents: List[Document]):
        self.documents = documents

        self.titles = [d.title for d in documents]
        self.content = [d.text for d in documents]
        self.vocab_frame = self.build_vocab(self.documents)

        self.order_centroids = None
        self.terms = None
        self.dist = None
        self.tfidf_matrix = None
        self.clusters = None


    @staticmethod
    def build_vocab(tech_news: List[Document]):
        stemmed_text = []
        tokenized_text = []
        for tn in tech_news:
            stemmed_text.extend(tokenize_and_stem(tn.text))
            tokenized_text.extend(tokenize(tn.text))

        return pd.DataFrame({'words': tokenized_text}, index=stemmed_text)

    def get_cluster_words(self, cluster, n=6):
        words = []
        for ind in self.order_centroids[cluster, :n]:
            words.append(self.vocab_frame.loc[self.terms[ind].split(' ')].values.tolist()[0][0])
        return ", ".join(words)


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


def calculate_tf_idf_matrix(corpus: CorpusMetadata):

    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.8,
        max_features=200000,
        min_df=0.2,
        stop_words='english',
        use_idf=True,
        tokenizer=tokenize_and_stem,
        ngram_range=(1, 3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus.content)

    corpus.terms = tfidf_vectorizer.get_feature_names()

    corpus.tfidf_matrix = tfidf_matrix
    corpus.dist = 1 - cosine_similarity(tfidf_matrix)


def k_means_clustering(corpus: CorpusMetadata):

    km = KMeans(n_clusters=NUMBER_OF_CLUSTERS)
    km.fit(corpus.tfidf_matrix)

    corpus.clusters = km.labels_.tolist()
    print("Clusters: {}".format(corpus.clusters))

    frame = pd.DataFrame({"title": corpus.titles, "cluster": corpus.clusters},
                         index=[corpus.clusters], columns=["title", "cluster"])
    print("Number of movies per cluster: \n{}".format(frame["cluster"].value_counts()))

    print("Top terms per cluster:\n")
    corpus.order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    for i in range(NUMBER_OF_CLUSTERS):
        print("Cluster {} words: {}".format(i, corpus.get_cluster_words(i)))

        print("Cluster {} titles:".format(i), end='')
        for title in frame.loc[i]['title'].values.tolist():
            print(" {},".format(title), end='')
        print("\n")


def visualize_clustering(corpus: CorpusMetadata):

    cluster_colors = {}
    for c_i in range(NUMBER_OF_CLUSTERS):
        cluster_colors[c_i] = '#%02X%02X%02X' % (rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255))

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(corpus.dist)
    xs, ys = pos[:, 0], pos[:, 1]

    cluster_names = dict([(i, corpus.get_cluster_words(i, 5)) for i in range(NUMBER_OF_CLUSTERS)])

    df = pd.DataFrame(dict(x=xs, y=ys, label=corpus.clusters, title=corpus.titles))

    groups = df.groupby('label')

    fig, ax = plt.subplots(figsize=(34, 18))
    ax.margins(0.05)

    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                label=cluster_names[name], color=cluster_colors[name], mec='none')
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')

    ax.legend(numpoints=1)

    for i in range(len(df)):
        ax.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['title'], size=8)

    plt.show()
    plt.close()


if __name__ == "__main__":

    tech_news_data = load_tech_news()

    corpus = CorpusMetadata(tech_news_data)

    calculate_tf_idf_matrix(corpus)
    k_means_clustering(corpus)
    visualize_clustering(corpus)
