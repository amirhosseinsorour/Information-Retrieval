from Phase1.PositionalBooleanSearch import MainDictionary, Word
from Phase1.PositionalBooleanSearch import load_dictionary as load_positional_dictionary
from Phase1.PositionalBooleanSearch import preprocess
import numpy as np
import operator
import pickle
import time


class Term:
    def __init__(self, word: Word):
        self.term = word.term
        self.df = len(word.postings)
        self.postings = self.convert_postings(word.postings)
        self.champions = self.make_champion_list(self.postings, r=100)

    @staticmethod
    def convert_postings(pos_postings):
        postings = {}
        for doc_id in pos_postings:
            tf = len(pos_postings[doc_id])
            postings[doc_id] = tf
        return postings

    @staticmethod
    def make_champion_list(postings_list, r):
        champions = sorted(postings_list.items(), key=operator.itemgetter(1), reverse=True)[:r]
        return {champions[i][0]: champions[i][1] for i in range(len(champions))}

    def tfidf_weight(self, doc_id, N):
        tf = self.postings[doc_id]
        return tfidf(tf, self.df, N)


class Dictionary:
    def __init__(self, pos_dict: MainDictionary):
        self.dictionary = self.convert_dict(pos_dict.get_dict())
        self.docs_titles = pos_dict.documents_titles
        self.N = len(self.docs_titles)
        self.docs_lengths = self.compute_lengths()

    @staticmethod
    def convert_dict(pos_dict):
        dictionary = {}
        for term in pos_dict:
            dictionary[term] = Term(pos_dict[term])
        return dictionary

    def compute_lengths(self):
        lengths = np.zeros(self.N)
        for term in self.dictionary:
            df = self.dictionary[term].df
            for doc_id, tf in self.dictionary[term].postings.items():
                weight = tfidf(tf, df, self.N)
                lengths[doc_id] += weight ** 2
        return np.sqrt(lengths)

    def search_query(self, query, champion):
        try:
            term = self.dictionary[query]
            if champion:
                return term.df, term.champions
            return term.df, term.postings
        except KeyError:
            return 0., {}

    def get_doc(self, doc_id):
        return self.docs_titles[doc_id]

    def get_term(self, term):
        return self.dictionary[term]


def tfidf(tf, df, N):
    if tf * df == 0:
        return 0.
    return (1 + np.log10(tf)) * np.log10(1. * N / df)


def cosine_score(query, dictionary, k, champion=False):
    scores = {}
    lengths = dictionary.docs_lengths
    # print(lengths)
    N = dictionary.N

    query_terms = preprocess([query])[0]
    # print(query_terms)
    for qt in query_terms:
        qt_df, qt_postings = dictionary.search_query(qt, champion)
        # print(qt_df)
        qt_tf = query_terms.count(qt)
        qt_weight = tfidf(qt_tf, qt_df, N)

        for doc_id, doc_tf in qt_postings.items():
            doc_weight = tfidf(doc_tf, qt_df, N)
            try:
                scores[doc_id] += qt_weight * doc_weight
            except KeyError:
                scores[doc_id] = qt_weight * doc_weight

    for doc_id in scores:
        # print(scores[doc_id])
        scores[doc_id] /= lengths[doc_id]
        # print(scores[doc_id])

    return retrieve_docs(scores, dictionary, k)


def retrieve_docs(scores, dictionary, k):
    scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    results = []
    for doc_id, score in scores:
        doc = dictionary.get_doc(doc_id)
        if doc not in results:
            results.append(doc)
        if len(results) == k:
            break
    return results


def save_dictionary(main_dictionary):
    with open('..\\Phase2\\dictionary.pkl', 'wb') as output:
        pickle.dump(main_dictionary, output)


def load_dictionary():
    with open('..\\Phase2\\dictionary.pkl', 'rb') as input:
        return pickle.load(input)


if __name__ == '__main__':
    # positional_dictionary = load_positional_dictionary()
    # main_dictionary = Dictionary(positional_dictionary)
    # save_dictionary(main_dictionary)

    main_dictionary = load_dictionary()

    while True:
        print(">> Please Enter your Query: ", end='')
        query = input()

        if not query:
            break

        start_time = time.time()
        print(">> Top 10 Results for «%s» : " % query)
        print("====================")
        for result in cosine_score(query, main_dictionary, k=10, champion=True):
            print(result.strip())
        print("===================================================")
        print(">> Retrieval Time: --- %s seconds ---" % (time.time() - start_time))
        print("===================================================")
