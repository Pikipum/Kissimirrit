import re
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

documents = ["This is a silly example",
             "A better example",
             "Nothing to see here",
             "This is a great and long example"]

cv = CountVectorizer(lowercase=True, binary=True)
sparse_matrix = cv.fit_transform(documents)
dense_matrix = sparse_matrix.todense()

td_matrix = dense_matrix.T  # .T transposes the matrix

terms = cv.get_feature_names()
sparse_td_matrix = sparse_matrix.T.tocsr()


d = {"and": "&", "AND": "&",
         "or": "|", "OR": "|",
         "not": "1 -", "NOT": "1 -",
         "(": "(", ")": ")"}  # operator replacements
t2i = cv.vocabulary_  # shorter notation: t2i = term-to-index

def main():


    corpus = open("corpus/wikicorpus.txt", "r")
    articles_str = ""
    for line in corpus:
        if re.search(r'<article name="', line):
           no_tags = re.sub(r'<article name="',"", line)
           no_tags_2 = re.sub(r'">', "", no_tags)
           articles_str += no_tags_2
        else:
           articles_str += line
    articles = articles_str.split("</article>")	
        

    #print("First term (with row index 0):", terms[0])
    #print("Third term (with row index 2):", terms[2])

    #print("\nterm -> IDX mapping:\n")
    #print(cv.vocabulary_)  # note the _ at the end

    #print("Row index of 'example':", cv.vocabulary_["example"])
    #print("Row index of 'silly':", cv.vocabulary_["silly"])


    #print("Query: example")
    #print(td_matrix[t2i["example"]])

    # Operators and/AND, or/OR, not/NOT become &, |, 1 -
    # Parentheses are left untouched
    # Everything else interpreted as a term and fed through td_matrix[t2i["..."]]



    def rewrite_token(t):
        return d.get(t, 'td_matrix[t2i["{:s}"]]'.format(t))  # Can you figure out what happens here?

    def rewrite_query(query):  # rewrite every token in the query
        return " ".join(rewrite_token(t) for t in query.split())

    def test_query(query):
        print("Query: '" + query + "'")
        print("Rewritten:", rewrite_query(query))
        print("Matching:", eval(rewrite_query(query)))  # Eval runs the string as a Python command
        print()

    test_query("example AND NOT nothing")
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

    test_query("NOT example OR great")

    hits_matrix = eval(rewrite_query("NOT example OR great"))
    print("Matching documents as vector (it is actually a matrix with one single row):", hits_matrix)
    print("The coordinates of the non-zero elements:", hits_matrix.nonzero())

    hits_list = list(hits_matrix.nonzero()[1])
    print(hits_list)

    for doc_idx in hits_list:
        print("Matching doc:", documents[doc_idx])

    for i, doc_idx in enumerate(hits_list):
        print("Matching doc #{:d}: {:s}".format(i, documents[doc_idx]))


main()
