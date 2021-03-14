Instructions on how to use the search engine:
 
 
First you must clone the repository. Most of our work is in the master branch, so you will need to specify the branch:
 
```
git clone -b master https://github.com/Pikipum/Kissimirrit.git
 
```
 
 
After you are done cloning, move to the flask folder, where our Twitter Corpus search engine is located. The name of the file is project.py
 
```
cd Kissimirrit/flask/
 
```
 
 
You will need to install several modules for the program to work. These are:
 
```
Regex, Sklearn, Numpy, NLTK, Flask, Matplotlib, Io, Csv, Langid, Pke, Os, spacy
 
```
 
 
To install pke from github, use the commands:
 
```
pip install git+https://github.com/boudinfl/pke.git
 
```
 
or 
 
```
python -m pip install git+https://github.com/boudinfl/pke.git
 
```
 
 
Pke will need some external resources, which can be installed using the commands:
 
 
 
```
python -m nltk.downloader stopwords
python -m nltk.downloader universal_tagset
python -m spacy download en
```
 
NLTK additionally requires the punkt module, which can be installed using the command:
 
 
```
python -m nltk.downloader punkt
 
```
 
 
After all of the required modules have been installed, it is time to setup the flask web server.
 
If you are using PyCharm, you can uncomment the last line of  the program (#app.run('127.0.0.1', debug=True)) to run the server inside of PyCharm.
 
To run the server from a terminal, you will need to set some environment variables. Change directory to Kissimirrit/flask/ and set up these variables:
 
```
export FLASK_APP=project.py
export FLASK_ENV=development
export FLASK_RUN_PORT=8000
 
```
 
 
After the variables have been set, the web server can be launched with:
 
 
```
flask run
 
```
 
Now you should be able to access the search engine from your browser with the link
 
```
 
localhost:8000/search
 
```
 
 
To use the search engine, simply type a word you want to find in the corpus, and it will show you the tweets and their relevancy scores and some graphs to the right, which contain information on where the tweets were posted from and relevance of keywords.
The search engine support single word queries, multiple word queries and exact matches with quotations marks, (e.g. “trump”).
 
In addition, it is possible to search for results in a specific language. This can be done by selecting a language from the dropdown menu “Languages”, clicking the desired language, and clicking the Languages button again. Now searching for a query will only bring results based on the selected language. Searching for the acronym "bts", for example, the search engine gives you completely different results if you have chosen Korean as the language, instead of English.
