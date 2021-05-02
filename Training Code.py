
import subprocess
import sys 

try:
    import nltk
    import keras
    import tensorflow as tf
    import tensorflow_hub as hub
    import sentencepiece
    import docx2pdf
    import PyPDF2
    #import python-docx
    
    
except ImportError:
     
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'nltk'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'keras'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'tensorflow'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'tensorflow_hub'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'sentencepiece'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'docx2pdf'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'PyPDF2'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'python-docx'])
    
finally:
    import os
    import tokenization
    import nltk
    import keras
    import tensorflow as tf
    import tensorflow_hub as hub
    import sentencepiece
    import docx2pdf
    import PyPDF2
    from docx2pdf import convert
    import json
    import datetime
    from nltk.stem.lancaster import LancasterStemmer
    stemmer = LancasterStemmer()
    import csv
    import numpy as np
    import pandas as pd
    import docx
    from docx import Document
    from nltk.corpus import stopwords
    from keras.preprocessing.text import Tokenizer 
    from keras.models import Sequential 
    from pandas import DataFrame 
    from matplotlib import pyplot as plt
    from keras import backend as K
    import string 
    import re 
    from os import listdir 
    from numpy import array
    from keras.preprocessing.sequence import pad_sequences 
    from keras.utils.vis_utils import plot_model
    from keras.layers import Flatten, Embedding, Dense 
    from keras.layers.convolutional import Conv1D, MaxPooling1D 
    import logging
    from keras.models import load_model
    from pickle import load
    from keras import *
    from keras.layers import SpatialDropout1D as SDropout
    from keras.layers import Dropout
################## BOW Definitions

def define_BOW_model(n_words):
  # define network 
    model = Sequential() 
    model.add(Dense(80, input_shape=(n_words,), activation='softmax' ))
    model.add(Dropout(0.2)) 
    model.add(Dense(50,input_shape=(70,), activation='softmax' ))
    model.add(Dropout(0.2)) 
    model.add(Dense(1, activation='sigmoid' ))
    # compile network 
    model.compile(loss='binary_crossentropy' , optimizer = "adam", metrics=['accuracy'])
    model.summary() 
    return model

def evaluate_mode(Xtrain, ytrain): # Xtest, ytest
  scores = list() 
  n_repeats = 1
  n_words = Xtrain.shape[1] 
  for i in range(n_repeats): 
    # define network 
    model_BOW = define_BOW_model(n_words) 
    # fit network 
    model_BOW.fit(Xtrain, ytrain, epochs=30,validation_split = 0.2, verbose=2,shuffle = True) 
    # evaluate 
    # what, acc = model_BOW.evaluate(Xtest, ytest, verbose=1)
    # print(what)
    # scores.append(acc) 
    # print(' %d accuracy: %s' % ((i+1), acc))
  return model_BOW #scores

def prepare_data(train_docs,mode): #test_docs
  global tokenizer_bow 
  # create the tokenizer 
  # fit the tokenizer on the documents 
  #print("Train docs")
  #print(train_docs)
  tokenizer_bow.fit_on_texts(train_docs) 
  #print("tokenizer after fit on texts")
  #print(tokenizer_bow)
  # encode training data set 
  Xtrain = tokenizer_bow.texts_to_matrix(train_docs, mode=mode) 
  #print("tokenizer after texts to matrix mode =  count")
  #print(Xtrain)
  # encode training data set 
  # Xtest = tokenizer_bow.texts_to_matrix(test_docs, mode=mode) 
  return Xtrain #Xtest

def train_bow_model():
  ytrain = np.array(y_train)
  # ytest = np.array(y_test)
  # run experiment 
  # modes = ['binary' , 'count' , 'tfidf' , 'freq' ] 
  modes = ['count'] 
  # results = DataFrame() 
  for mode in modes: 
    # prepare data for mode 
    Xtrain = prepare_data(X_train, mode) 
    # evaluate model on data for mode 
    model_BOW = evaluate_mode(Xtrain, ytrain)# Xtest, ytest


  from pickle import dump
  model_BOW.save("bow_model.h5")
  dump(tokenizer_bow, open('bow_tokenizer.pkl', 'wb'))

################### CNN Definitions

def train_cnn_model():
  ytrain = np.array(y_train)
  tokenizer_cnn = create_tokenizer(X_train)
  vocab_size = len(tokenizer_cnn.word_index) + 1 
  max_length = max([len(s.split()) for s in X_train])

  Xtrain = encode_docs(tokenizer_cnn, max_length, X_train)
  model_CNN = define_CNN_model(vocab_size, max_length)
  model_CNN.fit(Xtrain,ytrain, epochs=40, verbose=2,validation_split=0.2,shuffle=True)

  model_CNN.save("cnn_model.h5")
  from pickle import dump
  dump(tokenizer_cnn, open('cnn_tokenizer.pkl', 'wb'))
  return max_length

# Fit a tokenizer 
def create_tokenizer(lines): 
  tokenizer = Tokenizer() 
  tokenizer.fit_on_texts(lines) 
  return tokenizer

# Integer encode and pad documents 
def encode_docs(tokenizer, max_length, train_docs): 
  # integer encode 
  encoded = tokenizer.texts_to_sequences(train_docs) 
  # pad sequences 
  padded = pad_sequences(encoded, maxlen=max_length, padding='post') 
  return padded

# Define the model 
def define_CNN_model(vocab_size, max_length):
  model = Sequential()
  model.add(Embedding(vocab_size, 100, input_length=max_length)) 
  model.add(Conv1D(filters=32, kernel_size=3, activation='relu' ))
  model.add(SDropout(0.2))
  model.add(MaxPooling1D(pool_size=2)) 
  model.add(Flatten()) 
  model.add(Dense(10, activation='softmax',))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='sigmoid')) 
  # compile network 
  model.compile(loss='binary_crossentropy' , optimizer="Adam" ,run_eagerly=True, metrics=['accuracy']) 
  # summarize defined model 
  model.summary() 
  return model



########################## BERT Definitions

def train_bert_model():
  train_encode = bert_encode(X_train,tokenizer,max_len=bert_max_length)
  test_encode = bert_encode(X_test,tokenizer,max_len=bert_max_length)
  train_labels = np.array(y_train).reshape(len(y_train),1)

  max_len = bert_max_length
  model_bert = build_model(bert_layer, max_len=max_len)
  model_bert.summary()

  checkpoint = tf.keras.callbacks.ModelCheckpoint('model_bert.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
  earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, verbose=1)
  train_history = model_bert.fit(
      train_encode, train_labels, 
      validation_split=0.1,
      epochs=3,
      callbacks=[checkpoint, earlystopping],
      batch_size=32,
      verbose=1)

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def build_model(bert_layer, max_len=512):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(32, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(net)
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy']) 
    return model

################### Shared Definitions
def create_training_data():
  raw_data = []
  with open("Train-val-dataset.csv",'r',encoding = "utf8",errors = "ignore") as data: 
      for line in csv.DictReader(data):
          raw_data.append(line)
          #print(line)
      
  # adding key page to dictionaries
  for i in raw_data:
      i["page"] = None
      
  #Spliting zeroes and ones: reqs includes ones and not_reqs includes zeroes
  reqs = [] 
  not_reqs = []
  for i in raw_data:
      if i['Requirement'] == "0":
          not_reqs.append(i)
      else:
          reqs.append(i)
  #converting labels from text to integer
  for i in reqs:
      i["Requirement"] = int(i["Requirement"])
  for i in not_reqs:
      i["Requirement"] = int(i["Requirement"])   

  # seperating sentences and lables from other data and making list of lists
  list_reqs = [[i['Sentence'],i["Requirement"]] for i in reqs]
  list_not_reqs = [[i['Sentence'],i["Requirement"]] for i in not_reqs]

  # Divide the Data into Training and Test Set
  X_train = [i for i,j in list_reqs] + [i for i,j in list_not_reqs]
  y_train = [j for i,j in list_reqs] + [j for i,j in list_not_reqs]
  # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
  return X_train, y_train



X_train, y_train = create_training_data()
max_length = max([len(s.split()) for s in X_train])
tokenizer_bow = Tokenizer()
train_bow_model()
train_cnn_model()
#train_bert_model()