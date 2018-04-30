import os
import nltk
import math
from nltk.stem import WordNetLemmatizer
from nltk.corpus import brown
from stop_words import get_stop_words
from collections import Counter


#old 

#nltk.download('brown')
#categ = 'fiction'
#words = brown.words(categories=categ)

#tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

path_doc = "documents"
documents = {}
documents_tok = {}
documents_count = {} # tf
idf = {}
tfidf = {}
global_count = {}
set_words = set()

# Mot qui ne sont pas important
stop_words = get_stop_words('fr')

print("Chargement du corpus")
# Chargement du corpus 
for name_doc in os.listdir(path_doc):
	file = open(path_doc + "/" + name_doc, 'r')
	documents[name_doc] = file.read()
	file.close()


print("Tokenization du corpus")
# Tokenize les documents
for key, value in documents.items():
	words_tmp = nltk.word_tokenize(value)
	words_tmp = [w.lower() for w in words_tmp]
	words_tmp = [word for word in words_tmp if word not in stop_words]
	documents_tok[key] = words_tmp 
	set_words.update(words_tmp)


print("Calcul de la fréquence brut des mots dans chaque document (TF)")
# Comptage des mots dans les différents documents : Term Frequency
for key, value in documents_tok.items():
	documents_count[key] =  Counter(documents_tok[key])


print("Calcul de IDF des mots")
# Calcul du IDF des mots
for word in set_words:
	occur_doc = 0
	for key, value in documents_tok.items():
		if word in value:
			occur_doc += 1
	idf[word]= math.log10(len(documents) / occur_doc)

'''
# Comptage des mots de manière globale (compilation de tous les dictionnaires précédents) : peut etre pas utile
for value in documents_count.values():
	for key_word, count_word in value.items():
		if key_word in global_count:
			global_count[key_word] += count_word
		else:
			global_count[key_word] = count_word
'''




#lemmatizer = WordNetLemmatizer()
#tokens = [lemmatizer.lemmatize(token) for token in tokens]