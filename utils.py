#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:14:52 2018

@author: dhingratul
"""
import gensim
from nltk.stem import WordNetLemmatizer
import numpy as np
import matplotlib.pyplot as plt
import csv


def preProcess(corpus):
    """
    Helper function to preprocess text data with simple_preprocess using gensim
    Input: {corpus}
    -- corpus = corpus from intermediate step of create_data(string)
    Output: {preProcess}
    -- result = pre-processed corpus 
    Note:
        Please run from within create_data()
    """
    return gensim.utils.simple_preprocess(corpus, deacc=True, min_len=3)


def lemmatize_pos(text):
    """
    Helper function to Remove POS tags with gensim Lemmatize
    Input: {text}
    -- text = String of words
    Output: {result}
    -- result = List with POS tags starting with 'J' and 'V' removed
    Note:
        Please run from within create_data()
    """
    lemmas = gensim.utils.lemmatize(text)
    pos = [x.split("/")[-1] for x in lemmas]
    mask = [(x.startswith('J') or x.startswith('V'))for x in pos]
    mask_ = [not i for i in mask]
    words = [x.split("/")[0] for x in lemmas]
    result = [x for x, y in zip(words, mask_) if y == True]
    return result


def create_data(files, lemmat):
    """
    Helper function to create data from .tsv files by appling simple_preprocess using gensim and
    lemmatize(optional)
    Input: {files, lemmat}
    -- files = List of input file directories using glob.glob
    -- lemmat = Boolean {True, False} To enabale lemmatize by removing pos starting with J or V
    Output: {corpus_}
    -- corpus_ = Corpus of words as a list of list, with each list containig processed words from each
     document
    """
    corpus_ = []
    if lemmat is True:
        for filename in files:
            corpus_raw = ""
            with open(filename) as tsvfile:
                tsvreader = csv.reader(tsvfile, delimiter="\t")
                for line in tsvreader:
                    corpus_raw += line[-1]
                corpus_.append(lemmatize_pos(" ".join(preProcess(corpus_raw))))

    else:
        for filename in files:
            corpus_raw = ""
            with open(filename) as tsvfile:
                tsvreader = csv.reader(tsvfile, delimiter="\t")
                for line in tsvreader:
                    corpus_raw += line[-1]
                corpus_.append((preProcess(corpus_raw)))
    return corpus_


def process_texts(texts, stops, bigram_):
    """
    Helper Function to process text by Stopword Removal, Collocation detection and
    nltk Lemmatization 
    Input: {texts, stops, bigram_}
    -- texts = Tokenized text
    -- stops = NLTK stopword list
    -- bigram_ = bigram on corpus with gensim Phraser
    Output: {texts}
    -- texts = Pre-processed tokenized texts.
    """
    texts = [[word for word in line if word not in stops] for line in texts]
    texts = [bigram_[line] for line in texts]
    lemmatizer = WordNetLemmatizer()
    texts = [[word for word in lemmatizer.lemmatize(' '.join(line), pos='v').split()] for line in texts]
    return texts


def evaluate_bar_graph(coherences, indices):
    """
    Function to plot bar graph.

    coherences: list of coherence values
    indices: Indices to be used to mark bars. Length of this and coherences should be equal.
    """
    assert len(coherences) == len(indices)
    n = len(coherences)
    x = np.arange(n)
    plt.bar(x, coherences, width=0.2, tick_label=indices, align='center')
    plt.xlabel('Models')
    plt.ylabel('Coherence Value')
