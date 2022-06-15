import string

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import json
import string
import pandas as pd
import re

from tqdm import tqdm
from nltk.collocations import TrigramCollocationFinder, TrigramAssocMeasures
from unidecode import unidecode
import spacy
from nltk.stem import WordNetLemmatizer
nlp = WordNetLemmatizer()


def print_hi(name):
	# Use a breakpoint in the code line below to debug your script.
	print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatization(file):
	words = []
	for t in file:
		words.append(nlp.lemmatize(t))
	return " ".join(words)

# code taken from https://medium.com/analytics-vidhya/data-preparation-and-text-preprocessing-on-amazon-fine-food-reviews-7b7a2665c3f4
def decontracted(phrase):
	phrase = re.sub(r"won't", "will not", phrase)
	phrase = re.sub(r"can't", "can not", phrase)
	phrase = re.sub(r"n\'t", " not", phrase)
	phrase = re.sub(r"\'re", " are", phrase)
	phrase = re.sub(r"\'s", " is", phrase)
	phrase = re.sub(r"\'d", " would", phrase)
	phrase = re.sub(r"\'ll", " will", phrase)
	phrase = re.sub(r"\'t", " not", phrase)
	phrase = re.sub(r"\'ve", " have", phrase)
	phrase = re.sub(r"\'m", " am", phrase)
	phrase = re.sub(r"<br />", " ", phrase)  # added to get rid of the line breaks

	return phrase

# source: https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/
def clean_doc(doc):
	# remove abbreviations and line breaks
	doc = decontracted(doc)
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [word.translate(table) for word in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [word for word in tokens if not word in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]

	return tokens


# load the document
col_list = ["review", "sentiment"]
text = pd.read_csv("IMDB Dataset.csv", usecols=col_list)
file = text["review"][0]
print(file)
# to lowercase:
file = unidecode(file.lower())
# clean the doc
tokens = clean_doc(file)
# extract lemmas:
lemma = []
for w in tokens:
	w = nlp.lemmatize(w, get_wordnet_pos(w))
	lemma.append(w)
print(lemma)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
	print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
