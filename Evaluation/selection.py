import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import numpy as np
import math
import re
import logging
from scipy import spatial
from sentence_transformers import SentenceTransformer 

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def getMaxNgram(ngram, threshold):
    for idx, n in enumerate(ngram):
        if n < threshold:
            return idx, n

def getSelected(ngram, dists, final_outputs, threshold):
    dists_sorted = sorted(dists)
    for dist in dists_sorted:
        if type(dists) == np.ndarray:
            if ngram[np.where(dists == dist)[0][0]] < threshold:
                return dist, ngram[np.where(dists == dist)[0][0]], final_outputs[np.where(dists == dist)[0][0]]
        else: 
            if ngram[dists.index(dist)] < threshold:
                return dist, ngram[dists.index(dist)], final_outputs[dists.index(dist)]

def preProcessing(sentences):
    if type(sentences) == str:
        sentences = re.sub(r'\W+', ' ', sentences)
    else:
        sentences = [re.sub(r'\W+', ' ', s) for s in sentences]
    return sentences

def getVector(sentence, final_outputs):
    model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    reference_embeddings = model.encode(preProcessing([sentence]))
    canidates_embeddings = model.encode(preProcessing(final_outputs))
    distances = spatial.distance.cdist(reference_embeddings, canidates_embeddings, "cosine")[0]
    return distances