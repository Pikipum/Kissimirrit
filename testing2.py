import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

booknames = nltk.corpus.gutenberg.fileids()
#print(booknames)

bookdata = list(nltk.corpus.gutenberg.raw(name) for name in booknames)
#print(bookdata[0])

print("There are", len(bookdata), "books in the collection:", booknames)

gv = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2")
g_matrix = gv.fit_transform(bookdata).T.tocsr()



def search_gutenberg(query_string):
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
        print("Doc #{:d} (score: {:.4f}): {:s}".format(i, score, booknames[doc_idx]))
    print()


search_gutenberg("pequod")
search_gutenberg("alice entertained harriet")
search_gutenberg("whale hunter")
search_gutenberg("oh thy lord cometh")
search_gutenberg("which book should i read")