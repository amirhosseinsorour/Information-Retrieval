from Phase1.PositionalBooleanSearch import preprocess
from Phase2.VectorizedSearch import tfidf
from gensim.models import Word2Vec
import multiprocessing
import numpy as np
import pandas as pd
import pickle


class Term:
    def __init__(self, term, doc_id):
        self.term = term
        self.df = 1
        self.tf_postings = {doc_id: 1}

    def update_freq(self, doc_id):
        if doc_id in self.tf_postings:
            tf = self.tf_postings[doc_id]
            self.tf_postings[doc_id] = tf + 1
        else:
            self.tf_postings[doc_id] = 1
            self.df += 1

    def tfidf_weight(self, doc_id, N):
        tf = self.tf_postings[doc_id]
        return tfidf(tf, self.df, N)


class EmbeddingDictionary:
    def __init__(self, tokenized_docs):
        self.N = len(tokenized_docs)
        self.dictionary = self.build(tokenized_docs)

    def build(self, tokenized_docs):
        dictionary = {}

        for doc_id in range(self.N):
            document = tokenized_docs[doc_id]
            for token in document:
                if token in dictionary:
                    term = dictionary[token]
                    term.update_freq(doc_id)
                    dictionary[token] = term
                else:
                    term = Term(token, doc_id)
                    dictionary[token] = term

        return dictionary

    def get_dictionary(self):
        with open('..\\Phase3\\embedding_dictionary.pkl', 'wb') as output:
            pickle.dump(self.dictionary, output)
        return self.dictionary

    @staticmethod
    def load_dictionary():
        with open('..\\Phase3\\embedding_dictionary.pkl', 'rb') as input:
            return pickle.load(input)


def read_documents():
    contents = []
    topics = []

    df = pd.read_excel("..\\Phase3\\IR00_3_11k_News.xlsx")
    contents += df['content'].tolist()
    topics += df['topic'].tolist()

    df = pd.read_excel("..\\Phase3\\IR00_3_17k_News.xlsx")
    contents += df['content'].tolist()
    topics += df['topic'].tolist()

    df = pd.read_excel("..\\Phase3\\IR00_3_20k_News.xlsx")
    contents += df['content'].tolist()
    topics += df['topic'].tolist()

    return contents, topics


def prepare_training_data():
    documents, topics = read_documents()
    print(len(documents))  # 50061
    documents_tokens = preprocess(documents)
    with open('..\\Phase3\\train_tokenized_documents.pkl', 'wb') as output:
        pickle.dump(documents_tokens, output)
    with open('..\\Phase3\\train_tags.pkl', 'wb') as output:
        pickle.dump(topics, output)


def load_tokenized_docs():
    with open("..\\Phase3\\train_tokenized_documents.pkl", 'rb') as input:
        return pickle.load(input)


def build_model():
    training_data = load_tokenized_docs()
    cores = multiprocessing.cpu_count()

    w2v_model = Word2Vec(min_count=1, window=5, vector_size=300, alpha=0.03, workers=cores - 1)
    w2v_model.build_vocab(training_data)

    w2v_model.train(training_data, total_examples=w2v_model.corpus_count, epochs=20)
    w2v_model.save("w2v_model.model")


def build_docs_tfidf_dict():
    tokenized_docs = load_tokenized_docs()
    dictionary = EmbeddingDictionary(tokenized_docs).get_dictionary()
    N = len(tokenized_docs)
    docs_tfidf = []

    for doc_id in range(N):
        tokens = tokenized_docs[doc_id]
        doc_tfidf = {}
        for token in tokens:
            term = dictionary[token]
            doc_tfidf[token] = term.tfidf_weight(doc_id, N)
        docs_tfidf.append(doc_tfidf)

    return docs_tfidf


def documents_embedding():
    docs_embedding = []
    docs_tfidf = build_docs_tfidf_dict()
    model = Word2Vec.load("w2v_model.model")

    for document in docs_tfidf:
        doc_vector = np.zeros(300)
        weights_sum = 0

        for term, weight in document.items():
            weights_sum += weight
            try:
                doc_vector += model.wv[term] * weight
            except KeyError:
                continue

        if weights_sum == 0:
            docs_embedding.append(doc_vector)
            continue

        docs_embedding.append(doc_vector / weights_sum)

    return docs_embedding


def save_docs_vectors():
    docs_vectors = documents_embedding()
    with open('..\\Phase3\\train_docs_vectors.pkl', 'wb') as output:
        pickle.dump(docs_vectors, output)


def query_embedding(query, model, dictionary):
    query_terms = preprocess([query])[0]
    query_vector = np.zeros(300)
    N = len(dictionary)
    weights_sum = 0

    for qt in query_terms:
        try:
            qt_tf = query_terms.count(qt)
            qt_df = dictionary[qt].df
            weight = tfidf(qt_tf, qt_df, N)
            query_vector += model.wv[qt] * weight
        except KeyError:
            # print(qt)
            continue
        weights_sum += weight

    if weights_sum == 0:
        return query_vector

    return query_vector


if __name__ == '__main__':
    prepare_training_data()

    build_model()

    save_docs_vectors()
