import re, sklearn
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk


def main():
    corpus = open("corpus/wikicorpus.txt", "r", encoding='UTF-8')

    articles_str = ""
    for line in corpus:
        if re.search(r'<article name="', line):
            no_tags = re.sub(r'<article name="', "", line)
            no_tags_2 = re.sub(r'">', "", no_tags)
            articles_str += no_tags_2
        else:
            articles_str += line

    global articles
    articles = articles_str.split("</article>")

    corpus_with_names = {}
    for article in articles:
        lines = article.split('\n')
        if article == articles[0]:
            corpus_with_names[lines[0]] = ''.join(lines[1:])
        else:
            corpus_with_names[lines[1]] = ''.join(lines[2:])

    articles.pop()

    global articlenames, gv, g_matrix
    articlenames = list(corpus_with_names.keys())
    articledata = list(corpus_with_names[name] for name in articlenames)

    gv = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2")
    g_matrix = gv.fit_transform(articledata).T.tocsr()

    cv = CountVectorizer(lowercase=True, binary=True)
    cv._validate_vocabulary()
    sparse_matrix = cv.fit_transform(articles)
    binary_dense_matrix = cv.fit_transform(articles).T.todense()
    dense_matrix = cv.fit_transform(articles).T.todense()

    global terms
    terms = cv.get_feature_names()

    global sparse_td_matrix
    sparse_td_matrix = sparse_matrix.T.tocsr()

    global d
    d = {"and": "&", "AND": "&",
         "or": "|", "OR": "|",
         "not": "1 -", "NOT": "1 -",
         "(": "(", ")": ")"}  # operator replacements

    global t2i
    t2i = cv.vocabulary_

    inp = input("Search for a document: ")  # asks user for input
    while inp != '':
        if check_for_unknown_words(inp) == True:
            retrieve_articles(inp)
            search_wikicorpus(inp)
            inp = input("Search for another document: ")
        else:
            inp = input("Search for another document: ")


def check_for_unknown_words(query):
    tokens = query.split()
    for t in tokens:
        if t not in terms and t not in d.keys():
            print('Word "{}" is not found in corpus'.format(t))
            #return ([0 0 0 0]) <-- add len(articles)
            return False
    return True


def rewrite_query(query):  # rewrite every token in the query
    return " ".join(rewrite_token(t) for t in query.split())


def rewrite_token(t):
    return d.get(t, 'sparse_td_matrix[t2i["{:s}"]].todense()'.format(t))


def retrieve_articles(inp):

    hits_matrix = eval(rewrite_query(inp))  # feeds the user input into rewriting
    hits_list = list(hits_matrix.nonzero()[1])

    for i, doc_idx in enumerate(hits_list):
        if doc_idx == 0:
            lines = articles[doc_idx].split("\n")
            print("Matching doc #{:d}: {:s}\n {:s}\n".format(i, lines[0], lines[1]))
        else:
            lines = articles[doc_idx].split("\n")
            print("Matching doc #{:d}: {:s}\n {:s}\n".format(i, lines[1], lines[2]))

def search_wikicorpus(query_string):
    # Vectorize query string
    query_vec = gv.transform([query_string]).tocsc()

    # Cosine similarity
    hits = np.dot(query_vec, g_matrix)

    # Rank hits
    ranked_scores_and_doc_ids = \
        sorted(zip(np.array(hits[hits.nonzero()])[0], hits.nonzero()[1]),
               reverse=True)

    # Output result
    print("Your query '{:s}' matches the following documents:".format(query_string))
    for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
        print("Doc #{:d} (score: {:.4f}): {:s}".format(i, score, articlenames[doc_idx]))
    print()

main()