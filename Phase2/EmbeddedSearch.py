from Phase1.PositionalBooleanSearch import read_documents, preprocess
from Phase2.VectorizedSearch import load_dictionary, tfidf, retrieve_docs
from Phase2.VectorizedSearch import Dictionary, Term
from gensim.models import Word2Vec
from numpy.linalg import norm
import numpy as np
import multiprocessing
import pickle
import time


def prepare_training_data():
    documents, titles = read_documents()
    # print(len(documents))  # 7562
    documents_tokens = preprocess(documents)
    with open('..\\Phase2\\training_data.pkl', 'wb') as output:
        pickle.dump(documents_tokens, output)


def load_training_data():
    with open('..\\Phase2\\training_data.pkl', 'rb') as input:
        return pickle.load(input)


def build_model():
    training_data = load_training_data()
    cores = multiprocessing.cpu_count()

    w2v_model = Word2Vec(min_count=1, window=5, vector_size=300, alpha=0.03, workers=cores - 1)
    w2v_model.build_vocab(training_data)

    w2v_model.train(training_data, total_examples=w2v_model.corpus_count, epochs=20)
    w2v_model.save("w2v_model_300.model")


def load_model(model):
    return Word2Vec.load(model)


def build_docs_tfidf_dict(dictionary):
    documents_tokens = load_training_data()
    docs_tfidf = []

    for doc_id in range(len(documents_tokens)):
        tokens = documents_tokens[doc_id]
        doc_tfidf = {}
        for token in tokens:
            term = dictionary.get_term(token)
            doc_tfidf[token] = term.tfidf_weight(doc_id, dictionary.N)
        docs_tfidf.append(doc_tfidf)

    return docs_tfidf


def documents_embedding(model, dictionary):
    docs_embedding = []
    docs_tfidf = build_docs_tfidf_dict(dictionary)

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


def build_documents_vectors(model, dictionary, file_name):
    with open("..\\Phase2\\" + file_name, 'wb') as output:
        pickle.dump(documents_embedding(model, dictionary), output)


def load_documents_vectors(file_name):
    with open("..\\Phase3\\" + file_name, 'rb') as input:
        return pickle.load(input)


def query_embedding(query, model, dictionary):
    query_terms = preprocess([query])[0]
    query_vector = np.zeros(300)
    weights_sum = 0

    for qt in query_terms:
        try:
            qt_tf = query_terms.count(qt)
            qt_df = dictionary.get_term(qt).df
            weight = tfidf(qt_tf, qt_df, dictionary.N)
            query_vector += model.wv[qt] * weight
        except KeyError:
            # print(qt)
            continue
        weights_sum += weight

    if weights_sum == 0:
        # print("shit")
        return query_vector

    return query_vector / weights_sum


def embedded_search(query_vector, docs_vectors, dictionary, k):
    similarity = {}
    # print(norm(query_vector))
    for doc_id in range(len(docs_vectors)):
        doc_vector = docs_vectors[doc_id]
        vectors_norm = norm(query_vector) * norm(doc_vector)
        if vectors_norm == 0:
            similarity[doc_id] = 0
            continue
        similarity_score = np.dot(query_vector, doc_vector) / vectors_norm
        similarity[doc_id] = (similarity_score + 1) / 2

    return retrieve_docs(similarity, dictionary, k)


if __name__ == '__main__':
    # prepare_training_data()
    # build_model()

    model_name = "w2v_model_300.model"
    d2v_file_name = "d2v_model_300.pkl"

    # model_name = "w2v_150k_hazm_300_v2.model"
    # d2v_file_name = "d2v_150k_hazm_300_v2.pkl"

    # model_name = "..\\Phase3\\w2v_model.model"
    # d2v_file_name = "7k_docs_vectors.pkl"

    model = load_model(model_name)
    dictionary = load_dictionary()

    # # For Phase 3
    # # {
    # model = load_model("..\\Phase3\\w2v_model.model") #Copy
    # dictionary = load_dictionary()
    # build_documents_vectors(model, dictionary, file_name= "7k_docs_vectors.pkl")
    # # }
    # exit()

    # build_documents_vectors(model, dictionary, d2v_file_name)
    docs_vectors = load_documents_vectors(d2v_file_name)

    while True:
        print(">> Please Enter your Query: ", end='')
        query = input()

        if not query:
            break

        query_vector = query_embedding(query, model, dictionary)

        start_time = time.time()
        print(">> Top 10 Results for «%s» : " % query)
        print("====================")
        for result in embedded_search(query_vector, docs_vectors, dictionary, k=10):
            print(result.strip())
        print("===================================================")
        print(">> Retrieval Time: --- %s seconds ---" % (time.time() - start_time))
        print("===================================================")
