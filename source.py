import os
import nltk
import math
import operator
import csv
import scipy
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

def tokenize_sentence(sentence):
    words_tmp = nltk.word_tokenize(sentence)
    words_tmp = [w.lower() for w in words_tmp] # lowercase
    words_tmp = [word for word in words_tmp if word not in stop_words] # remove stop_words
    return(words_tmp)


# Comptage des mots dans les différents documents : Term Frequency
def computeFrequency(documents_tok):
    documents_count = {}
    for key, value in documents_tok.items():
        documents_count[key] =  Counter(documents_tok[key])
    return(documents_count)


# Calcul du IDF des mots
def computeIDF(documents_tok, nb_doc):
    idf = {}
    for word in set_words:
        #tfidf_tot[word] = 0
        occur_doc = 0
        for key, value in documents_tok.items():
            if word in value:
                occur_doc += 1
        idf[word] = math.log10((len(documents) / occur_doc))
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


def save(sorted_tfidf, path='saveTFIDF_tot2.csv'):
    print("Sauvegarde du TFIDF")
    with open(path, 'w',  encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        for pair in sorted_tfidf:
           writer.writerow([pair[0], pair[1]])

def create_cooccurrence_matrix(filename,window_size):
    vocabulary={}
    data=[]
    row=[]
    col=[]
    text = open(filename,"r").read()
    text = text.replace("\n", "")
    for sentence in text.split("."):
        sentence=sentence.strip()
        tokens=[token for token in tokenize_sentence(sentence) if token!=u""]
        for pos,token in enumerate(tokens):
            i=vocabulary.setdefault(token,len(vocabulary))
            start=max(0,pos-window_size)
            end=min(len(tokens),pos+window_size+1)
            for pos2 in range(start,end):
                if pos2==pos: 
                    continue
                j=vocabulary.setdefault(tokens[pos2],len(vocabulary))
                data.append(1.); row.append(i); col.append(j);
    cooccurrence_matrix=scipy.sparse.coo_matrix((data,(row,col)))
    return vocabulary,cooccurrence_matrix
'''

# Comptage des mots de manière globale (compilation de tous les dictionnaires précédents) : peut etre pas utile
for value in documents_count.values():
    for key_word, count_word in value.items():
        if key_word in global_count:
            global_count[key_word] += count_word
        else:
            global_count[key_word] = count_word

https://stackoverflow.com/a/8685873 penser à citer
# read the csv
with open('dict.csv', 'rb') as csv_file:
    reader = csv.reader(csv_file)
    mydict = dict(reader)
'''




#lemmatizer = WordNetLemmatizer()
#tokens = [lemmatizer.lemmatize(token) for token in tokens]
'''
if __name__ == "__main__":
    print("Initialisation des variables")
    documents = {}
    documents_tok = {} 
    set_words = set()
    idf = {}
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
'''