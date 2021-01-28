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

    query = "example OR nothing"

    def check_for_unknown_words(query):

        tokens = query.split()
        for t in tokens:
            if t not in terms and t not in d.keys():
                print('Word "{}" is not found in corpus'.format(t))
                return False
        return True


    def rewrite_token(t):
        return d.get(t, 'td_matrix[t2i["{:s}"]]'.format(t))

    def rewrite_query(query):
        return " ".join(rewrite_token(t) for t in query.split())

    def test_query(query):
        print("Query: '" + query + "'")
        print("Rewritten:", rewrite_query(query))
        print("Matching:", eval(rewrite_query(query)))
        print()


    if check_for_unknown_words(query) == True:
        test_query(query)
        #test_query("NOT example OR great")
        #test_query("( NOT example OR great ) AND nothing")  # AND, OR, NOT can be written either in ALLCAPS
        #test_query("( not example or great ) and nothing")  # ... or all small letters
        #test_query("not example and not nothing")

    #print(sparse_matrix)
    #print(sparse_matrix.tocsc())
    #print(sparse_matrix.T)


    #print(sparse_td_matrix)

    def rewrite_token(t):
        return d.get(t, 'sparse_td_matrix[t2i["{:s}"]].todense()'.format(t))

    #test_query("NOT example OR great")

    hits_matrix = eval(rewrite_query("NOT example OR great"))
    #print("Matching documents as vector (it is actually a matrix with one single row):", hits_matrix)
    #print("The coordinates of the non-zero elements:", hits_matrix.nonzero())

    hits_list = list(hits_matrix.nonzero()[1])
    #print(hits_list)

    for doc_idx in hits_list:
        print("Matching doc:", documents[doc_idx])

    for i, doc_idx in enumerate(hits_list):
        print("Matching doc #{:d}: {:s}".format(i, documents[doc_idx]))


main()