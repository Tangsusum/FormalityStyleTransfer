from nltk import ngrams
from nltk import bigrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
import numpy as np
import os
import math
import re
import nltk
from textblob import TextBlob
import spacy 
import en_core_web_sm
from collections import defaultdict

"""
PINC score
"""
def PINCScore(source, candidate):

    def generate_ngrams(s, n):
        # Convert to lowercases
        s = s.lower()
        
        # Replace all none alphanumeric characters with spaces
        s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
        
        # Break sentence in the token, remove empty tokens
        tokens = [token for token in s.split(" ") if token != ""]
        
        # Use the zip function to help us generate n-grams
        # Concatentate the tokens into ngrams and return
        ngrams = zip(*[tokens[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    def preProcessing (sentences):
        sentences = [re.sub(r'\W+', ' ', _) for _ in sentences]
        sentences = [_.rstrip() for _ in sentences]
        return sentences

    def getSource (sentences,n):
        allNgrams = []
        for s in sentences:
            allNgrams = allNgrams + generate_ngrams(s,n)
        return list(dict.fromkeys(allNgrams))

    n = 3
    ngrams = []
    for j in range(1,n+1):
        count = 0
        for i in getSource(preProcessing(source),j): 
            if i in generate_ngrams(candidate,j):
                count += 1 
        ngrams.append(1 - abs(count/len(generate_ngrams(candidate,j))))
    return sum(ngrams)/len(ngrams)

"""
BLEU score - uses smoothing function 7 
Refer to https://www.aclweb.org/anthology/W14-3346/ for details
"""
def getBLEU(reference, candidate):

    bleu = sentence_bleu
    method = SmoothingFunction()
    # return bleu(reference, candidate, smoothing_function=method.method2)
    return bleu(reference, candidate, smoothing_function=method.method7)

"""
Ngram score
"""
def ngramsScore(source, candidate):

    def generate_ngrams(s, n):
        # Convert to lowercases
        s = s.lower()

        # Replace all none alphanumeric characters with spaces
        s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)

        # Break sentence in the token, remove empty tokens
        tokens = [token for token in s.split(" ") if token != ""]

        # Use the zip function to help us generate n-grams
        # Concatentate the tokens into ngrams and return
        ngrams = zip(*[tokens[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    def preProcessing(sentences):
        sentences = re.sub(r'\W+', ' ', sentences)
        sentences = sentences.lower()
        sentences = sentences.rstrip()
        return sentences

    n = 2
    ngrams = []
    for j in range(1, n+1):
        count = 0
        for i in generate_ngrams(preProcessing(source), j):
            if i in generate_ngrams(preProcessing(candidate), j):
                count += 1
        # ngrams.append(abs(count/len(generate_ngrams(candidate, j))))
        ngrams.append(abs(count/len(generate_ngrams(source, j))))
    return sum(ngrams)/len(ngrams)

"""
Formality Score
"""
def F_score(doc):

    nlp = en_core_web_sm.load()

    doc = nlp(doc)
    freq = defaultdict(lambda: 0 )
    dictionary = {
        'noun': ['NOUN'],
        'adj': ['ADJ'], 
        'prep': ['ADP'],
        'article': ['DET'],
        'pronoun': ['PRON'],
        'verb': ['VERB'],
        'adverb': ['ADV'],
        'interjection': ['INTJ']
    }
    
    for i in doc:
        for key, val in dictionary.items():
            if i.pos_ in val:
                freq[key] += 1

    F_score = (freq['noun'] + freq['adj'] + freq['prep'] + freq['article'] - freq['pronoun'] - freq['verb'] - freq['adverb'] - freq['interjection'] + 100)*0.5
    return F_score

"""
Adjective Density Formality (ADF) score
"""
def ADF_score(sentence):

    def Tokens (sentences):
        sentences = re.sub(r'\W+', ' ', sentences) 
        sentences = sentences.rstrip()
        tokens = list(dict.fromkeys(sentences.split()))
        return len(tokens)

    nlp = en_core_web_sm.load()

    doc = nlp(sentence)
    freq = defaultdict(lambda: 0 )
    dictionary = {
        'noun': ['NOUN'],
        'adj': ['ADJ'], 
        'prep': ['ADP'],
        'article': ['DET'],
        'pronoun': ['PRON'],
        'verb': ['VERB'],
        'adverb': ['ADV'],
        'interjection': ['INTJ']
    }
    
    for i in doc:
        for key, val in dictionary.items():
            if i.pos_ in val:
                freq[key] += 1

    ADFscore = (freq['adj']/Tokens(sentence))*100
    return ADFscore
