import re
import string
import tensorflow
import nltk
from sklearn.model_selection import RepeatedKFold, cross_val_score, StratifiedKFold, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
# from tensorflow.keras.models import save_model
from keras import saving
import numpy as np
import pandas as pd
from keras import layers
from keras import models
from keras.preprocessing.text import Tokenizer
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from numpy import array
from tensorflow.python.keras.saving.save import load_model
from unidecode import unidecode
import keras.backend as K

nlp = WordNetLemmatizer()
positive_lines = list()
negative_lines = list()
acc_per_fold = []
loss_per_fold = []


def count_loss_function(y_true, y_pred):
    casted = K.cast(y_pred, "int32")
    loss = 1.0
    # casted = tensorflow.cast(y_pred, tensorflow.int32)
    if K.equal(y_true, casted):
        loss = 0.0
    else:
        loss = 1.0
    return K.cast(loss, "float32")


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
            dic[word] = dic[word] + 1


def writeToFile(dictionary, name):
    dic = dict(sorted(dictionary.items(), reverse=True, key=lambda x: x[1]))
    for key in list(dic.keys()):
        with open(name + ".txt", "a") as f:
            print(key, '\t', file=f)
    f.close()


def buildVocabulary(text):
    dic = dict()
    for i in range(0, 3000):
        file = text["review"][i]
        tokens = clean_doc(file)
        lemmas = []
        for w in tokens:
            w = nlp.lemmatize(w, get_wordnet_pos(w))
            lemmas.append(w)
        dictionary(lemmas, dic)
    for i in range(25002, 28002):
        file = text["review"][i]
        tokens = clean_doc(file)
        lemmas = []
        for w in tokens:
            w = nlp.lemmatize(w, get_wordnet_pos(w))
            lemmas.append(w)
        dictionary(lemmas, dic)
    # remove word with low occurrence
    min_frequency = 2
    final_dic = dict()
    for word in dic:
        if dic[word] >= min_frequency:
            final_dic[word] = dic[word]
    writeToFile(final_dic, "vocabulary")


def process_docs(text, vocab):
    for i in range(0, 3000):
        file = text["review"][i]
        tokens = clean_doc(file)
        tokens = [w for w in tokens if w in vocab]
        line = ' '.join(tokens)
        negative_lines.append(line)
    for i in range(25002, 28002):
        file = text["review"][i]
        tokens = clean_doc(file)
        tokens = [w for w in tokens if w in vocab]
        line = ' '.join(tokens)
        positive_lines.append(line)


def process_test_data(text, vocab):
    for i in range(3001, 4001):
        file = text["review"][i]
        tokens = clean_doc(file)
        tokens = [w for w in tokens if w in vocab]
        line = ' '.join(tokens)
        negative_lines.append(line)
    for i in range(28003, 29003):
        file = text["review"][i]
        tokens = clean_doc(file)
        tokens = [w for w in tokens if w in vocab]
        line = ' '.join(tokens)
        positive_lines.append(line)


# load the document
col_list = ["review", "sentiment"]
text = pd.read_csv("IMDB Dataset.csv", usecols=col_list)
# buildVocabulary(text)
file = open("vocabulary.txt", 'r')
vocabulary = file.read()
file.close()
vocabulary = vocabulary.split()
vocabulary = set(vocabulary)
process_docs(text, vocabulary)
tokenizer = Tokenizer()
grid_param = {
    'n_estimators': [70, 150, 200],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}

# fit the tokenizer on the documents
docs = negative_lines + positive_lines
tokenizer.fit_on_texts(docs)
Xtrain = tokenizer.texts_to_matrix(docs, mode='freq')
ytrain = array([0 for _ in range(3000)] + [1 for _ in range(3000)])
positive_lines.clear()
negative_lines.clear()
process_test_data(text, vocabulary)
docs = negative_lines + positive_lines
Xtest = tokenizer.texts_to_matrix(docs, mode='freq')
print(Xtrain.shape)
ytest = array([0 for _ in range(1000)] + [1 for _ in range(1000)])
features = Xtest.shape[1]
kfold = KFold(n_splits=5, shuffle=True)
fold_no = 1
# for train, test in kfold.split(Xtrain, ytrain):
#     # Define the model architecture
#     network = models.Sequential()
#     network.add(layers.Dense(units=50, activation='relu', input_shape=(features,)))
#     network.add(layers.Dense(units=50, activation='relu'))
#     network.add(layers.Dense(units=1, activation='sigmoid'))
#
#     # Compile the model
#     network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     # Generate a print
#     print('------------------------------------------------------------------------')
#     print(f'Training for fold {fold_no} ...')
#
#     # Fit data to model
#     history = network.fit(Xtrain[train], ytrain[train],
#                           epochs=50,
#                           verbose=2)
#
#     # Generate generalization metrics
#     scores = network.evaluate(Xtrain[test], ytrain[test], verbose=0)
#     print(
#         f'Score for fold {fold_no}: {network.metrics_names[0]} of {scores[0]}; {network.metrics_names[1]} of {scores[1] * 100}%')
#     acc_per_fold.append(scores[1] * 100)
#     loss_per_fold.append(scores[0])
#     filepath = './saved_model/' + str(fold_no)
#     network.save(filepath = filepath)
#     # Increase fold number
#     fold_no = fold_no + 1
#
# # == Provide average scores ==
# print('------------------------------------------------------------------------')
# print('Score per fold')
# for i in range(0, len(acc_per_fold)):
#     print('------------------------------------------------------------------------')
#     print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
# print('------------------------------------------------------------------------')
# print('Average scores for all folds:')
# print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
# print(f'> Loss: {np.mean(loss_per_fold)}')
# print('------------------------------------------------------------------------')
# print(all_accuracies.std())
loaded_network = load_model('saved_model/4')
loss, acc = loaded_network.evaluate(Xtest, ytest, verbose=0)
print('accuracy: %f' % (acc * 100))
