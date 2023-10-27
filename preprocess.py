import load_data as ld
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

nltk.download('stopwords')

from nltk.corpus import stopwords
english_stopwords = stopwords.words('english')

def tokenize(sent):
    words = pos_tag(word_tokenize(sent))
    words = [w[0].lower() for w in words if ('NNG' in w[1] or 'NNP' in w[1])]
    # XR: root, NNP: proper noun, NNG: common noun, VA: adjective
    return words

def clean_text(str):
    txt = re.sub('[-=+,#/\?:^@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', str)
    return txt

def clean_stopword(tokens):
    clean_tokens = []
    for token in tokens:
        if token not in english_stopwords:
            clean_tokens.append(token)
    return clean_tokens

def preprocessing(paragraph):  # paragraph is a single document
    clean_txt = clean_text(paragraph)
    clean_tokens = tokenize(clean_txt)
    clean_tokens = clean_stopword(clean_tokens)

    return clean_tokens  # [token1, token2, token3, ...]

def stc_preprocessing(listOfstcs):
    return list(map(preprocessing, listOfstcs))

def para2stcs(paragraph):
    return paragraph.split('. ')

def lst2str(lst):
    return ' '.join(lst)

def preprocess_node(paragraphs):  # paragraphs: ["document1", "document2", ...]
    tokens_of_paras = list(map(preprocessing, paragraphs))
    return tokens_of_paras  # [[words of document1], [words of document2], ...]

def preprocess_edge(paragraphs):  # paragraphs: ["document1", "document2", ...]
    listOfsentences = list(map(para2stcs, paragraphs))      # [[sentence1 of document1, sentence2 of document1, ...], [sentence1 of document2, ...] , ...]
    clean_stc_tokens = list(map(stc_preprocessing, listOfsentences))

    return clean_stc_tokens

def make_dic_tfidf(list_of_wordlist):
    total_word_counts = dict()
    word_frequency = dict()
    for word_list in list_of_wordlist:
        word_counts = dict()
        for word in word_list:
            word_counts[word] = word_counts.get(word, 0) + 1    # Increase the word count
        
        for item in word_counts.items():
            total_word_counts[item[0]] = total_word_counts.get(item[0], 0) + item[1]

    # Not calculating tfidf, only tf. tfidf is commented out.
    tfidfs = []
    for item in word_frequency.items():
        tf = total_word_counts[item[0]]
        tfidfs.append(tf)

    tfidfs = np.array(tfidfs) 
    tfidfs = tfidfs / np.linalg.norm(tfidfs)
    for idx, item in enumerate(word_frequency.items()):
        total_word_counts[item[0]] = round(tfidfs[idx], 2)

    return total_word_counts

def make_dic_count(paragraphs):
    word_counts = dict()
    for word in paragraphs:
        word_counts[word] = word_counts.get(word, 0) + 1    # Increase the word count

    return word_counts

def stcs_dic_count(ListOfSentence):
    return list(map(make_dic_count, ListOfSentence))
    