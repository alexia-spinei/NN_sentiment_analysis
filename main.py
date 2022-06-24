import re
#import tenserflow
from keras.preprocessing.text import Tokenizer
import nltk
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import json
import string
import pandas as pd
from tqdm import tqdm
from nltk.collocations import TrigramCollocationFinder, TrigramAssocMeasures
from unidecode import unidecode

nlp = WordNetLemmatizer()
positive_lines = list()
negative_lines = list()


def count_loss_function(y_actual, y_predicted):
	if y_actual == y_predicted:
		loss_value = 0
	else:
		loss_value = 1
	return loss_value



def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatization(file):
	tokenized = nltk.word_tokenize(file)
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
	doc = unidecode(doc.lower())
	doc = decontracted(doc)
	# remove punctuation from each token
	for char in string.punctuation:
		doc = doc.replace(char, ' ')
	# split into tokens by white space
	tokens = doc.split()
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [word for word in tokens if not word in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]

	return tokens


def dictionary(file, dic):
	for word in file:
		if word not in dic:
			dic[word] = 1
		else:
			dic[word] = dic[word]+1


def writeToFile(dictionary,name):
	dic = dict(sorted(dictionary.items(), reverse=True, key = lambda x: x[1]))
	for key in list(dic.keys()):
		with open(name+".txt", "a") as f:
			print(key, '\t', dic[key], file = f)
	f.close()


def process_docs(text):
	for i in range(0,10):
		file = text["review"][i]
		tokens = clean_doc(file)
		line = ' '.join(tokens)
		if text["sentiment"][i] == "positive":
			positive_lines.append(line)
		else:
			negative_lines.append(line)

# load the document
col_list = ["review", "sentiment"]
text = pd.read_csv("IMDB Dataset.csv", usecols=col_list)
process_docs(text)
dic = dict()
lines = list()
for i in range(0,10):
	file = text["review"][i]
	tokens = clean_doc(file)
	line = ' '.join(tokens)
	lemmas = []
	for w in tokens:
		w = nlp.lemmatize(w, get_wordnet_pos(w))
		lemmas.append(w)
	dictionary(lemmas, dic)
# remove word with low occurrence
min_frequency = 2
final_dic = dict()
# for word in dic:
# 	if dic[word] >= min_frequency:
# 		final_dic[word] = dic[word]
#writeToFile(final_dic, "vocabulary")

tokenizer = Tokenizer()

# fit the tokenizer on the documents
docs = positive_lines + negative_lines
tokenizer.fit_on_texts(docs)
Xtrain = tokenizer.texts_to_matrix(docs, mode='freq')
print(Xtrain.shape)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
