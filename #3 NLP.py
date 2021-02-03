import re, sklearn
from sklearn.feature_extraction.text import CountVectorizer


def main():

    corpus = open("corpus/wikicorpus.txt", "r", encoding='UTF-8')
    articles_str = ""
    for line in corpus:
        if re.search(r'<article name="', line):
           no_tags = re.sub(r'<article name="',"", line)
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
    dense_matrix = sparse_matrix.todense()

    td_matrix = dense_matrix.T

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

    retrieve_articles()

def check_for_unknown_words(query):

    tokens = query.split()
    for t in tokens:
        if t not in terms and t not in d.keys():
            print('Word "{}" is not found in corpus'.format(t))
            return False
    return True


def rewrite_query(query):  # rewrite every token in the query

    return " ".join(rewrite_token(t) for t in query.split())

def test_query(query):

    print("Query: '" + query + "'")
    print("Rewritten:", rewrite_query(query))
    print("Matching:", eval(rewrite_query(query)))  # Eval runs the string as a Python command


    if check_for_unknown_words(query):
        test_query(query)
        #test_query("NOT example OR great")
        #test_query("( NOT example OR great ) AND nothing")  # AND, OR, NOT can be written either in ALLCAPS
        #test_query("( not example or great ) and nothing")  # ... or all small letters
        #test_query("not example and not nothing")

    print(sparse_matrix)
    print(sparse_matrix.tocsc())
    print(sparse_matrix.T)


    print(sparse_td_matrix)

def rewrite_token(t):
    
     return d.get(t, 'sparse_td_matrix[t2i["{:s}"]].todense()'.format(t))


def retrieve_articles():

    inp = input("Search for a document: ") # asks user for input
    
    if check_for_unknown_words(inp) == True:
        hits_matrix = eval(rewrite_query(inp)) # feeds the user input into rewriting
        hits_list = list(hits_matrix.nonzero()[1])

        for i, doc_idx in enumerate(hits_list):
           if doc_idx == 0:
               lines = articles[doc_idx].split("\n")
               print("Matching doc #{:d}: {:s}\n {:s}\n".format(i, lines[0], lines[1]))
           else:
               lines = articles[doc_idx].split("\n")
               print("Matching doc #{:d}: {:s}\n {:s}\n".format(i, lines[1], lines[2]))


main()
