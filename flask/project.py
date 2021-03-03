import re, sklearn
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from flask import Flask, render_template, render_template_string, request
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flask import Response
import matplotlib
import csv
import langid
import pke



def main():

    #read & process the corpus here
    global tweets
    tweets = []
    global languages
    languages = []
    tweetcorp = open("corpus/tweetcorpus.tsv", encoding="utf-8")
    read_tsv = csv.reader(tweetcorp, delimiter="\t")

    for row in read_tsv:
        tweets.append(row)
        if langid.classify(row[3])[0] not in languages:
            language = langid.classify(row[3])
            languages.append(language[0])

    tweetcorp.close()
 
    global lang_id_dict         # a dictionary with language codes as keys and tweet IDs as values
    lang_id_dict = {}
    for tweet in tweets:
        lang_id_dict[langid.classify(tweet[3])[0]] = tweet[0]

    global tweets_data, tweets_id
    tweets_data = []
    tweets_id = []
    for each in tweets:
        tweets_data.append(each[3])
        tweets_id.append(each[0])

 
    #Structure of tweets: print(tweets[0])
    #Tweet structure: Tweet ID [0], Country [1], Date [2], Tweet [3], and other parameters
    #[3] is the actual content of the tweet. To print out the 15th tweet of the corpus:
    #Use for example print(tweets[15][3])

    global stemmer
    stemmer = SnowballStemmer("english")
 
    
app = Flask(__name__)

main()

matplotlib.use('Agg')

@app.route('/search')

def search():


    global gv, g_matrix, terms
    gv = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2")
    g_matrix = gv.fit_transform(tweets_data).T.tocsr()
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

            words_known = check_for_unknown_words(each.strip('"').lower())     # Check if the token is in corpus,
            if words_known == False:                                                    # if it's not, stop loop & store the value as FALSE
                matches.append('Word "{}" is not found in corpus'.format(each))
                break

        if stemmed == True: # Stem the query
            stemmed_inp = " ".join(stemmer.stem(each) for each in inp.split()) # stems every word if query is a multi-word phrase
            inp = stemmed_inp

        if words_known:
            term = inp.split()
            gv.ngram_range = (len(term), len(term))
            g_matrix = gv.fit_transform(tweets_data).T.tocsr()
            search_wikicorpus(inp)

    return render_template('index.html', matches=matches, languages=languages)

@app.route('/select_language', methods=['POST', 'GET'])
def selected_language():
   global selected_language
   selected_language = str(request.form.get("selected_lang"))

   #return render_template_string('''<p>{{ selected_language }}</p>''', selected_language=selected_language)
   return(str(selected_language))

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
    tweet_results = open('results.txt', 'w')
    for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
        tweet_results.write(tweets_data[doc_idx])
        matches.append("Doc #{:d} (score: {:.4f}): {:s}\n".format(i, score, tweets_id[doc_idx]))
    tweet_results.close()

    keyphrases_and_scores = {} # dictionary with the keyphrases/themes as keys and their scores as values

    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(input='results.txt')
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyphrases = extractor.get_n_best(n=10)
    for keyphrase in keyphrases:
        keyphrases_and_scores[keyphrase[0]] = f'{keyphrase[1]:.5f}'
 

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


