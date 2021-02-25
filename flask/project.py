import re, sklearn
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flask import Response
import matplotlib

def main():

    #read & process the corpus here
 
    global stemmer
    stemmer = SnowballStemmer("english")


app = Flask(__name__)

matplotlib.use('Agg')

@app.route('/search')

def search():

    main()

    global gv, g_matrix, terms
    gv = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2")
    g_matrix = gv.fit_transform(both_data).T.tocsr()
    gv._validate_vocabulary()
    terms = gv.get_feature_names()

    global matches
    matches = []
    stemmed = True
    inp = request.args.get('query')
    if inp:
        both = ""
        for each in inp.split():
            if re.match('["][\w\s]+|[\w\s]+["]|["][\w\s]+["]', each): # Checks if input has quotation marks
                both += each.strip('"') + " "
                stemmed = False # Sets the input to search unstemmed documents (exact matches)
            else:
                both += stemmer.stem(each) + " "


            inp = both.strip()
            inp = re.sub('"', '', inp) # Removes quotation marks

            words_known = check_for_unknown_words(each.strip('"').lower(), stemmed)     # Check if the token is in corpus,
                if words_known == False:                                                    # if it's not, stop loop & store the value as FALSE
                    matches.append('Word "{}" is not found in corpus'.format(each))
                    break

        if stemmed == True: # Stem the query
            stemmed_inp = " ".join(stemmer.stem(each) for each in inp.split()) # stems every word if query is a multi-word phrase
            inp = stemmed_inp


        if words_known:
            gv.ngram_range = (len(term), len(term))
            g_matrix = gv.fit_transform(both_data).T.tocsr()
            search_wikicorpus(inp)

    return render_template('index.html', matches=matches)

def check_for_unknown_words(t):

    if t not in terms:
        return False
    return True


def search_wikicorpus(query_string):

    global ranked_scores_and_doc_ids
    query_vec = gv.transform([query_string]).tocsc() # Vectorize query string
    hits = np.dot(query_vec, g_matrix) # Cosine similarity
    ranked_scores_and_doc_ids = \
        sorted(zip(np.array(hits[hits.nonzero()])[0], hits.nonzero()[1]), reverse=True)


    # Output result
    for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
        matches.append("Doc #{:d} (score: {:.4f}): {:s}\n".format(i, score, both_names[doc_idx]))


@app.route('/test.png')
def plot_image():
    #Creates a plot and saves it in test.png.
    #Still need to show it in the HTML page.
    plot_articles = []
    plot_scores = []
    for score, name in ranked_scores_and_doc_ids:
        plot_articles.append(both_names[name])
        plot_scores.append(score)


    fig, ax = plt.subplots()
    #  ax = fig.add_axes([0,0,1,1])
    if len(plot_articles) > 5:
        ax.bar(plot_articles[0:5], plot_scores[0:5], color='purple')
    else:
        ax.bar(plot_articles, plot_scores, color='purple')
    fig.suptitle("Articles and their scores")
    png_image = io.BytesIO()
    FigureCanvas(fig).print_png(png_image)

    return Response(png_image.getvalue(), mimetype='image/png')


#app.run('127.0.0.1', debug=True)
