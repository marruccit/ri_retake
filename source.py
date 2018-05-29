import os
import nltk
import math
import operator
import csv
import scipy
import itertools
import pandas as pd
from scipy import spatial
from nltk.stem import WordNetLemmatizer
from nltk.corpus import brown
from stop_words import get_stop_words
from collections import Counter
from nltk.stem.snowball import FrenchStemmer


path_doc = "documents"
# Mot qui ne sont pas important
stop_words = get_stop_words('fr')
# Ajout de la ponctuation
stop_words +=  [")", "(", ".", ",", ";", ":", "!", "?", "/", "^", "'", "’", "\"", "~", "-", "_", "•",  "...", "..", "*", "**"]


# Chargement du corpus 
def loadCorpus(path=path_doc):
    documents = {}
    for name_doc in os.listdir(path_doc):
        file = open(path_doc + "/" + name_doc, 'r')
        documents[name_doc] = file.read()
        file.close()
    return(documents)

 # Tokenize les documents
def tokenize(documents):
    for key, value in documents.items():
        value = value.replace("\n", "")
        words_tmp = nltk.word_tokenize(value)
        words_tmp = [w.lower() for w in words_tmp] # lowercase
        words_tmp = [word for word in words_tmp if word not in stop_words] # remove stop_words
        documents_tok[key] = words_tmp
    return(documents_tok)



# Comptage des mots dans les différents documents : Term Frequency
def computeFrequency(documents_tok):
    documents_count = {}
    for key, value in documents_tok.items():
        documents_count[key] =  Counter(documents_tok[key])
    return(documents_count)


# Calcul du IDF des mots
def computeIDF(documents_tok, nb_doc, smooth=0):
    if (smooth > 1):
        smooth = 1
    idf = {}
    for word in set_words:
        #tfidf_tot[word] = 0
        occur_doc = 0
        for key, value in documents_tok.items():
            if word in value:
                occur_doc += 1
        idf[word] = math.log10(smooth + (len(documents) / occur_doc))
    return(idf)


# Calcul du tfidf
def computeTFIDF(documents_count, idf):
    for name_doc in documents_count.keys():
        print(name_doc)
        tfidf[name_doc] = {}
        for word in set_words:
            tfidf[name_doc][word] = documents_count[name_doc][word] * idf[word]
    return(tfidf)


def sumTFIDF(tfidf):
    tfidf_tot = {}
    for name_doc, tfidf_doc in tfidf.items():
        for word, value in tfidf_doc.items():
            if word in tfidf_tot.keys():
                tfidf_tot[word] += value
            else:
                tfidf_tot[word] = 0
    return(tfidf_tot)


def save(sorted_tfidf, path):
    print("Sauvegarde du TFIDF")
    with open(path, 'w',  encoding='utf-16') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(['word', 'value'])
        for pair in sorted_tfidf:
           writer.writerow([pair[0], pair[1]])


# Tokenize une phrase
def tokenize_sentence(sentence):
    words_tmp = nltk.word_tokenize(sentence)
    words_tmp = [w.lower() for w in words_tmp] # lowercase
    words_tmp = [word for word in words_tmp if word not in stop_words] # remove stop_words
    return(words_tmp)


# Créer la matrice de cooccurence sur les mots sélectionnés
# Deux mots coocurrent si ils sont dans la même phrase
def getCoocurrenceMatrix(documents, words):
    matrix = pd.DataFrame(0, index=words, columns=words)
    content_docs = '.\n'.join(documents.values())
    content_docs = content_docs.replace("\n", "")
    for sentence in content_docs.split("."):
        wordsOccur = []
        sentence=sentence.strip()
        tokens=[token for token in tokenize_sentence(sentence) if token!=u""]
        for token in tokens:
            if token in words:
                wordsOccur.append(token)
        if len(wordsOccur) > 1:
            listPairs = list(itertools.permutations(wordsOccur, 2))
            for pair in listPairs:
                matrix[pair[0]][pair[1]] += 1
    return(matrix)

# Calcul la distance cosinus entre chaque mots de la matrice
def getCosineDist(coo_matrix):
    words = coo_matrix.columns
    cosine_matrix = pd.DataFrame(0, index=words, columns=words)
    listPairs = list(itertools.permutations(words, 2))
    for pair in listPairs:
        cosine_matrix[pair[0]][pair[1]] = 1 - spatial.distance.cosine(coo_matrix[pair[0]].tolist(), coo_matrix[pair[1]].tolist())
    return cosine_matrix


def computeTFIDF(path_file='tfidf_tot.csv'):
    print("Initialisation des variables")
    documents = {}
    documents_tok = {} 
    set_words = set()
    idf = {}
    idf_smooth = {}
    tfidf = {}
    tfidf_tot = {}
    print("Chargement du corpus")
    documents = loadCorpus()
    
    nb_doc = len(documents)
    print("Tokenization du corpus")
    documents_tok = tokenize(documents)
    set_words.update([w for sublist in documents_tok.values() for w in sublist])
    print("Calcul de la fréquence brut des mots dans chaque document (TF)")
    documents_count = computeFrequency(documents_tok)
    
    print("Calcul de IDF des mots " + "("+ str(len(set_words)) + " mots)")
    idf = computeIDF(documents_tok, nb_doc)    
    print("Calcul du TFIDF des mots")
    tfidf = computeTFIDF(documents_count, idf)
    print("Aggrégation du TFIDF des mots")
    tfidf_tot = sumTFIDF(tfidf)
    print("Tri du TFIDF")
    sorted_tfidf = sorted(tfidf_tot.items(), key=operator.itemgetter(1))
    save(sorted_tfidf, path_file)


def computeSimilarity(path_file='tfidf_tot.csv'):
    documents = loadCorpus()
    words = getKeepWords(path_file)
    coo_matrix = getCoocurrenceMatrix(documents, words)
    cosine_matrix = getCosineDist(coo_matrix)