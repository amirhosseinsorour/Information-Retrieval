from Phase3.Embedding import query_embedding, EmbeddingDictionary, Term
from gensim.models import Word2Vec
import numpy as np
import pickle
import random
import time

TAGS = ['sport', 'economy', 'political', 'culture', 'health']


class Class:
    def __init__(self, tag):
        self.tag = tag
        self.documents = []
        self.doc_ids = []

    def add_vector(self, vector, doc_id):
        self.documents.append(vector)
        self.doc_ids.append(doc_id)


class KNN:
    def __init__(self, vectors, k=50):
        self.vectors = vectors
        self.classes = {tag: Class(tag) for tag in TAGS}
        self.k = k

    @staticmethod
    def load_train_data():
        with open("..\\Phase3\\train_docs_vectors.pkl", 'rb') as input1:
            with open("..\\Phase3\\train_tags.pkl", 'rb') as input2:
                return pickle.load(input1), pickle.load(input2)

    @staticmethod
    def cosine_similarity(vector1, vector2):
        vectors_norm = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if vectors_norm == 0:
            return 0.

        return np.dot(vector1, vector2) / vectors_norm

    @staticmethod
    def find_most_tag(tag_list):
        for t in range(len(tag_list)):
            if tag_list[t] == 'sports':
                tag_list[t] = 'sport'
            if tag_list[t] == 'politics':
                tag_list[t] = 'political'

        res, count = '', 0
        for tag in TAGS:
            if tag_list.count(tag) > count:
                count = tag_list.count(tag)
                res = tag
        return res

    def run(self):
        training_data, train_tags = self.load_train_data()

        for doc_id in range(len(self.vectors)):
            vector = self.vectors[doc_id]
            similarities = [self.cosine_similarity(vector, doc) for doc in training_data]
            doc_ids = np.argpartition(similarities, -self.k)[-self.k:]
            tag = self.find_most_tag([train_tags[id] for id in doc_ids])
            self.classes[tag].add_vector(vector, doc_id)

    def save_classes(self):
        with open('..\\Phase3\\classes.pkl', 'wb') as output:
            pickle.dump(self.classes, output)

    @staticmethod
    def load_classes():
        with open('..\\Phase3\\classes20.pkl', 'rb') as input:
            return pickle.load(input)


def run_classification():
    with open('..\\Phase3\\7k_docs_vectors.pkl', 'rb') as input:
        vectors = pickle.load(input)

    knn = KNN(vectors)
    knn.run()
    knn.save_classes()


def retrieve_docs(doc_ids, news, k):
    results = []
    for doc_id in doc_ids:
        doc = news[doc_id]
        if doc not in results:
            results.append(doc)
        if len(results) == k:
            break
    return results


def search_classes(query_vector, category: Class, news, num=10):
    documents = category.documents
    doc_ids = category.doc_ids

    similarities = [KNN.cosine_similarity(query_vector, doc) for doc in documents]
    sorted_doc_ids = [doc_ids[index] for index in
                      sorted(range(len(similarities)), key=lambda n: similarities[n], reverse=True)]

    return retrieve_docs(sorted_doc_ids, news, k=num)


def decompose_query(raw_query):
    cat = random.choice(TAGS)
    for tag in TAGS:
        if tag in raw_query:
            cat = tag

    query = raw_query.replace("cat:" + cat, "")
    return query, cat


def load_news():
    with open('..\\Phase3\\7k_news_titles.pkl', 'rb') as input:
        return pickle.load(input)


if __name__ == '__main__':
    # run_classification()
    # exit(2)

    classes = KNN.load_classes()
    news = load_news()
    model = Word2Vec.load("w2v_model.model")
    dictionary = EmbeddingDictionary.load_dictionary()

    while True:
        print(">> Please Enter your Query: ", end='')
        query = input()

        if not query:
            break

        query, cat = decompose_query(query)
        query_vector = query_embedding(query, model, dictionary)

        start_time = time.time()
        print(">> Top 10 Results for «%s» : " % query)
        print("====================")
        for result in search_classes(query_vector, classes[cat], news, num=10):
            print(result.strip())
        print("===================================================")
        print(">> Retrieval Time: --- %s seconds ---" % (time.time() - start_time))
        print("===================================================")
