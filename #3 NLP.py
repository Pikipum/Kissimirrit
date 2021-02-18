import re, sklearn
from sklearn.feature_extraction.text import CountVectorizer


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
    articles.pop()

    cv = CountVectorizer(lowercase=True, binary=True)
    cv._validate_vocabulary()
    sparse_matrix = cv.fit_transform(articles)
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
        if check_for_unknown_words(inp):
            retrieve_articles(inp)
            inp = input("Search for another document: ")
        else:
            inp = input("Search for another document: ")


def check_for_unknown_words(query):
    tokens = query.split()
    for t in tokens:
        if t not in terms and t not in d.keys():
            print('Word "{}" is not found in corpus'.format(t))
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


main()