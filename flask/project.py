import re, sklearn
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from flask import Flask, render_template, request, url_for, redirect
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flask import Response
import matplotlib
import csv
import langid
import pke
import os


def main():

    # read & process the corpus here
    global tweets
    tweets = []
    global languages, countries
    languages = []
    countries = []
    tweetcorp = open("corpus/tweetcorpus.tsv", encoding="utf-8")
    read_tsv = csv.reader(tweetcorp, delimiter="\t")

    for row in read_tsv:
        tweets.append(row)
        if row[1] not in countries:
            countries.append(row[1])
        if langid.classify(row[3])[0] not in languages:
            language = langid.classify(row[3])
            languages.append(language[0])

    languages = sorted(languages)          # Sort language options for webpage
    #languages.insert(0,'en')
    tweetcorp.close()

    global stemmer_dict
    stemmer_dict =  {'en':'english', 'da':'danish', 'nl':'dutch', 'fi':'finnish', 'fr':'french',            # Maps langid language abbreviations to
                     'de':'german', 'hu':'hungarian', 'it':'italian', 'no':'norwegian',                     # SnowballStemmer arguments
                     'pt':'portuguese','ro':'romanian', 'ru':'russian', 'es':'spanish', 'sv':'swedish'}

    global lang_tweet_dict, lang_id_dict, id_date_dict, id_og_dict
    lang_tweet_dict = {lang: [] for lang in languages}  # Create 2 dictionaries with empty list for each language to use as dataset
    lang_id_dict = {lang: [] for lang in languages}
    id_date_dict = {}                                   # Map tweet ids to country & date of the tweet
    id_og_dict ={}                                      # Preserve original (unstemmed/unjoined) tweets
    for each in tweets[1:]:
        tweet = re.sub(r'http.*', r'', each[3])         # Clean the tweet
        if each[4] == '1':
            language = 'en'
        else:
            language = langid.classify(tweet)[0]
        if language in stemmer_dict.keys():             # Check if the language is supported by SnowballStemmer
            stemmer = SnowballStemmer(stemmer_dict[language])
            tokens = nltk.word_tokenize(tweet)
            stemmed_tweet = ' '.join(stemmer.stem(t) for t in tokens)
            lang_tweet_dict[language].append(tweet + stemmed_tweet)
        else:
            lang_tweet_dict[language].append(tweet)
        lang_id_dict[language].append(each[0])
        id_date_dict[each[0]] = each[1]+' '+each[2]
        id_og_dict[each[0]] = tweet
        #tweets_data.append(each[3] + stemmed_tweet)
        #tweets_id.append(each[0])
    global selected_language                   # Set English as default language
    selected_language = 'en'


    #global lang_id_dict  # a dictionary with language codes as keys and tweet IDs as values
    #lang_id_dict = {}
    #for tweet in tweets:
     #   lang_id_dict[langid.classify(tweet[3])[0]] = tweet[0]

    #global tweets_data, tweets_id
    #tweets_data = []
    #tweets_id = []
    #for each in tweets:
     #   tokens = nltk.word_tokenize(each[3])
      #  stemmed_tweet = ' '.join(stemmer.stem(t) for t in tokens)
       # tweets_data.append(each[3] + stemmed_tweet)
       # tweets_id.append(each[0])

    # Structure of tweets: print(tweets[0])
    # Tweet structure: Tweet ID [0], Country [1], Date [2], Tweet [3], and other parameters
    # [3] is the actual content of the tweet. To print out the 15th tweet of the corpus:
    # Use for example print(tweets[15][3])

IMAGES_FOLDER = os.path.join('static', 'images')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGES_FOLDER

main()

matplotlib.use('Agg')


@app.route('/search')
def search():
    full_filename_gif = os.path.join(app.config['UPLOAD_FOLDER'], 'twitter_bird.gif')
    full_filename_png = os.path.join(app.config['UPLOAD_FOLDER'], 'twitter_image.png')

    if selected_language != 'en':
        languages.pop(languages.index(selected_language))
        languages.pop(languages.index('en'))
        languages.sort()
        languages.insert(0, selected_language)
        languages.insert(1, 'en')
    elif selected_language == 'en':
        languages.pop(languages.index('en'))
        languages.sort()
        languages.insert(0, 'en')
    
    

    global tweets_data, tweets_id
    tweets_data = lang_tweet_dict[selected_language]
    tweets_id = lang_id_dict[selected_language]

    global gv, g_matrix, terms
    gv = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2")
    g_matrix = gv.fit_transform(tweets_data).T.tocsr()
    gv._validate_vocabulary()
    terms = gv.get_feature_names()

    global matches
    matches = []
    words_known = False
    error = ""
    stemmed = True
    plot_2 = selected_language == 'en'        # Draw themes plot only if language is English
    inp = request.args.get('query')
    if inp:
        both = ""
        for each in inp.split():
            if re.match('["][\w\s]+|[\w\s]+["]|["][\w\s]+["]', each) or selected_language not in stemmer_dict.keys():  # Checks if input has quotation marks and is stemmable
                both += each.strip('"') + " "
                stemmed = False  # Sets the input to search unstemmed documents (exact matches)
            else:
                stemmer = SnowballStemmer(stemmer_dict[selected_language])
                both += stemmer.stem(each) + " "

            inp = both.strip()
            inp = re.sub('"', '', inp)  # Removes quotation marks

            words_known = check_for_unknown_words(each.strip('"').lower())  # Check if the token is in corpus,
            if words_known == False:  # if it's not, stop loop & store the value as FALSE
                if selected_language in stemmer_dict.keys():    # Inform user which language was used as corpus
                    language = stemmer_dict[selected_language][0].upper() + stemmer_dict[selected_language][1:]
                else:
                    language = selected_language.upper()
                error = 'Word "{}" is not found in {} corpus.'.format(each, language)
                break

        if stemmed == True:  # Stem the query
            stemmed_inp = " ".join(stemmer.stem(each) for each in inp.split())  # stems every word if query is a multi-word phrase
            inp = stemmed_inp

        if len(inp.split()) > 1:
            term = inp.split()
            gv.ngram_range = (len(term), len(term))
            g_matrix = gv.fit_transform(tweets_data).T.tocsr()
            multiword_terms = gv.get_feature_names()
            if inp not in multiword_terms:
                error = f'Phrase "{inp}" is not found in corpus.'
                words_known = False
            else:
                words_known = True

        if words_known:
            term = inp.split()
            gv.ngram_range = (len(term), len(term))
            g_matrix = gv.fit_transform(tweets_data).T.tocsr()
            search_wikicorpus(inp)

    return render_template('index.html', matches=matches, languages=languages, countries=countries,
                           words_known=words_known, error=error, plot_2=plot_2, full_filename_gif=full_filename_gif,
                           full_filename_png=full_filename_png, selected_language=selected_language)


@app.route('/select_language', methods=['POST', 'GET'])
def select_language():
    global selected_language
    selected_language = str(request.form.get("selected_lang"))
    return redirect(url_for('search'))


def check_for_unknown_words(t):
    if t not in terms:
        return False
    return True


def search_wikicorpus(query_string):
    global ranked_scores_and_doc_ids
    query_vec = gv.transform([query_string]).tocsc()  # Vectorize query string
    hits = np.dot(query_vec, g_matrix)  # Cosine similarity
    ranked_scores_and_doc_ids = \
        sorted(zip(np.array(hits[hits.nonzero()])[0], hits.nonzero()[1]), reverse=True)

    # Output result
    with io.open('results.txt', 'w', encoding='UTF-8') as tweet_results:
        matches.append('Relevance ranking:\n')
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            #tweet_results.write(id_og_dict[tweets_id[doc_idx]])
            tweet_results.write(tweets_data[doc_idx])
            matches.append("#{:d} (score: {:.4f}): {}\n".format(i, score, id_og_dict[tweets_id[doc_idx]]))
    tweet_results.close()

    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(input='results.txt')
    extractor.candidate_selection()
    extractor.candidate_weighting()

    global keyphrases
    keyphrases = extractor.get_n_best(n=10)
    keyphrases_and_scores = {}  # dictionary with the keyphrases/themes as keys and their scores as values
    for keyphrase in keyphrases:
        keyphrases_and_scores[keyphrase[0]] = f'{keyphrase[1]:.5f}'


@app.route('/test.png')
def plot_image():
    # Creates a plot and saves it in test.png.
    # Still need to show it in the HTML page.
    plot_tweets = []
    plot_scores = []
    for score, tweet_id in ranked_scores_and_doc_ids:
        plot_tweets.append('\n'.join(id_date_dict[tweets_id[tweet_id]].split()))    #Show tweet country & date (instead of id nr.)
        plot_scores.append(score)

    fig, ax = plt.subplots()
    #  ax = fig.add_axes([0,0,1,1])
    if len(plot_tweets) > 5:
        ax.bar(plot_tweets[0:5], plot_scores[0:5], color='purple')
    else:
        ax.bar(plot_tweets, plot_scores, color='purple')
    fig.suptitle("Country and date of relevant tweets")
    png_image = io.BytesIO()
    FigureCanvas(fig).print_png(png_image)

    return Response(png_image.getvalue(), mimetype='image/png')


@app.route('/test2.png')
def plot_keyphrase_image():
    plot_words = []
    plot_keyphrase_scores = []
    for word, score in keyphrases:
        plot_words.append(word)
        plot_keyphrase_scores.append(score)

    fig, ax = plt.subplots()
    if len(plot_words) > 5:
        ax.bar(plot_words[0:5], plot_keyphrase_scores[0:5], color='purple')
    else:
        ax.bar(plot_words, plot_keyphrase_scores, color='purple')
    fig.suptitle("Keywords and their relevance")
    png_image = io.BytesIO()
    FigureCanvas(fig).print_png(png_image)

    return Response(png_image.getvalue(), mimetype='image/png')

app.run('127.0.0.1', debug=True)
