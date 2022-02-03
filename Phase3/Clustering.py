from Phase3.Embedding import query_embedding, EmbeddingDictionary, Term
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
import pickle
import time


class Cluster:
    def __init__(self, centroid):
        self.centroid = centroid
        self.documents = []
        self.doc_ids = []

    def add_vector(self, vector, doc_id):
        self.documents.append(vector)
        self.doc_ids.append(doc_id)

    def mean(self):
        return np.mean(self.documents, axis=0)

    def rss(self):
        return sum([np.linalg.norm(doc - self.centroid) ** 2 for doc in self.documents])


class KMeans:
    def __init__(self, vectors, k=100, epochs=10):
        self.vectors = vectors
        self.centroids = random.sample(vectors, k)
        self.clusters = [Cluster(c) for c in self.centroids]
        self.cluster_num = k
        self.epochs = epochs
        self.G = []  # RSS over time

    @staticmethod
    def cosine_similarity(vector1, vector2):
        vectors_norm = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if vectors_norm == 0:
            return 0.

        return np.dot(vector1, vector2) / vectors_norm

    def rss(self):
        return sum([c.rss() for c in self.clusters])

    def run(self):
        for e in range(self.epochs):
            print("epoch = ", e + 1)

            for doc_id in range(len(self.vectors)):
                vector = self.vectors[doc_id]
                similarities = [self.cosine_similarity(vector, cluster.centroid) for cluster in self.clusters]
                cluster_index = similarities.index(max(similarities))
                self.clusters[cluster_index].add_vector(vector, doc_id)

            # Measurement
            self.G.append(self.rss())

            # Last epoch
            if e == self.epochs - 1:
                break

            # update centroids
            self.centroids = [c.mean() for c in self.clusters]
            self.clusters = [Cluster(c) for c in self.centroids]

    def plot_rss(self):
        e = [i for i in range(self.epochs)]

        plt.plot(e, self.G, color='green')
        plt.xlabel("Epoch")
        plt.ylabel("RSS")
        plt.show()

    def save_clusters(self):
        with open('..\\Phase3\\clusters.pkl', 'wb') as output:
            pickle.dump(self.clusters, output)

    @staticmethod
    def load_clusters():
        with open('..\\Phase3\\clusters100.pkl', 'rb') as input:
            return pickle.load(input)


def load_vectorized_docs():
    with open('..\\Phase3\\train_docs_vectors.pkl', 'rb') as input:
        return pickle.load(input)


def run_clustering():
    docs_vectors = load_vectorized_docs()

    k_means = KMeans(docs_vectors)
    k_means.run()
    k_means.plot_rss()
    k_means.save_clusters()


def retrieve_docs(doc_ids, news, k):
    results = []
    for doc_id in doc_ids:
        doc = news[doc_id]
        if doc not in results:
            results.append(doc)
        if len(results) == k:
            break
    return results


def search_clusters(query_vector, clusters, news, b=1, num=10):
    cluster_similarities = [KMeans.cosine_similarity(query_vector, cluster.centroid) for cluster in clusters]
    cluster_indexes = np.argpartition(cluster_similarities, -b)[-b:]

    all_selected_docs = []
    all_selected_doc_ids = []
    for i in range(b):
        all_selected_docs += clusters[cluster_indexes[i]].documents
        all_selected_doc_ids += clusters[cluster_indexes[i]].doc_ids

    similarities = [KMeans.cosine_similarity(query_vector, doc) for doc in all_selected_docs]
    sorted_doc_ids = [all_selected_doc_ids[index] for index in
                      sorted(range(len(similarities)), key=lambda n: similarities[n], reverse=True)]

    return retrieve_docs(sorted_doc_ids, news, k=num)


def map_id():
    titles = []

    df = pd.read_excel("..\\Phase3\\IR00_3_11k_News.xlsx")
    titles += df['url'].tolist()

    df = pd.read_excel("..\\Phase3\\IR00_3_17k_News.xlsx")
    titles += df['url'].tolist()

    df = pd.read_excel("..\\Phase3\\IR00_3_20k_News.xlsx")
    titles += df['url'].tolist()

    return titles


def load_news():
    with open('..\\Phase3\\50k_news_titles.pkl', 'rb') as input:
        return pickle.load(input)


if __name__ == '__main__':
    # run_clustering()
    # exit(2)

    clusters = KMeans.load_clusters()
    news = load_news()
    model = Word2Vec.load("w2v_model.model")
    dictionary = EmbeddingDictionary.load_dictionary()

    while True:
        print(">> Please Enter your Query: ", end='')
        query = input()

        if not query:
            break

        query_vector = query_embedding(query, model, dictionary)

        start_time = time.time()
        print(">> Top 10 Results for «%s» : " % query)
        print("====================")
        for result in search_clusters(query_vector, clusters, news, b=1, num=10):
            print(result.strip())
        print("===================================================")
        print(">> Retrieval Time: --- %s seconds ---" % (time.time() - start_time))
        print("===================================================")
