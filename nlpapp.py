import nltk, re, pprint
from nltk import word_tokenize
from urllib import request
from bs4 import BeautifulSoup

def main():

        url = "https://www.hs.fi/kotimaa/art-2000007759719.html"
        html = request.urlopen(url).read().decode('utf8')
        raw = BeautifulSoup(html, 'html.parser').get_text()
        tokens = word_tokenize(raw)
        soup = BeautifulSoup(html, 'html.parser')
        print(soup.prettify())



main()
