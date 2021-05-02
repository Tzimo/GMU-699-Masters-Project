# This is the code to run an original documents through and get several different
# Predictions about whether each sentence in the document is a requirement.

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
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
################## BERT Definitions

# Function to encode the sentences for the model that uses BERT using a tokenizer
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


################### CNN Definitions

# integer encode and pad documents 
def encode_docs(tokenizer, max_length, train_docs): 
  # integer encode 
  encoded = tokenizer.texts_to_sequences(train_docs) 
  # pad sequences 
  padded = pad_sequences(encoded, maxlen=max_length, padding='post') 
  return padded


################### Shared Definitions
# Slice the original document into sentences. Must finally be in PDF format.
def slicing_original_documents_pdf(filename):
  pdfobj = open(filename,'rb')
  pdfreader = PyPDF2.PdfFileReader(pdfobj)
  page_counts = pdfreader.numPages
  final_text = []
  originaldoc_data12 = {}
  pages = 0
  for i in range(page_counts):
      pages += 1
      page_obj = pdfreader.getPage(i)
      page_txt = page_obj.extractText()
      page_txt = page_txt.replace('\n','')
      period_count = page_txt.count(".")
      if period_count != 0:
          for j in range(period_count+1):
              period_idx = page_txt.find(".")
              if period_idx != len(page_txt)-1 and period_idx != -1:
                  if page_txt[period_idx-1] == "e" and page_txt[period_idx+1] == "g":
                      page_txt = page_txt.replace(page_txt[period_idx],"-",1)
                  elif page_txt[period_idx-1] == "g" and page_txt[period_idx-2] == "." and page_txt[period_idx-3] == "e":
                      page_txt = page_txt.replace(page_txt[period_idx],"-",1)
                  elif page_txt[period_idx-1] == "i" and page_txt[period_idx+1] == "e":
                      page_txt = page_txt.replace(page_txt[period_idx],"-",1)
                  elif page_txt[period_idx-1] == "." or page_txt[period_idx+1] == ".":
                      page_txt = page_txt.replace(page_txt[period_idx],"-",1)
                  elif page_txt[period_idx-1].isalpha() and (page_txt[period_idx+1].isspace() or page_txt[period_idx+1].isalpha() or page_txt[period_idx+1] == " "):
                      corpus_slice = slice(period_idx+1)
                      if page_txt[corpus_slice] != '' or page_txt[corpus_slice] !=  '':
                        final_text.append([page_txt[corpus_slice],pages])
                        originaldoc_data12[page_txt[corpus_slice]]=pages
                        page_txt = page_txt.replace(page_txt[0:period_idx+1]," ")
                  elif page_txt[period_idx-1] == ")" and (page_txt[period_idx+1].isspace() or page_txt[period_idx+1].isalpha()):
                      corpus_slice = slice(period_idx+1)
                      if page_txt[corpus_slice] != '' or page_txt[corpus_slice] !=  '':
                        final_text.append([page_txt[corpus_slice],pages])
                        originaldoc_data12[page_txt[corpus_slice]]=pages
                        page_txt = page_txt.replace(page_txt[0:period_idx+1]," ")
                  elif page_txt[period_idx-1].isnumeric() and (page_txt[period_idx+1].isnumeric() or page_txt[period_idx+1].isalpha()):
                      page_txt = page_txt.replace(page_txt[period_idx],"-",1)
                  elif page_txt[period_idx-1].isalpha() and page_txt[period_idx+1].isnumeric():
                      page_txt = page_txt.replace(page_txt[period_idx],"-",1)
                  elif page_txt[period_idx-1].isnumeric() and (page_txt[period_idx+1].isnumeric() or page_txt[period_idx+1] == " "):
                      page_txt = page_txt.replace(page_txt[period_idx],"-",1)
                  else:
                      page_txt = page_txt.replace(page_txt[period_idx],"-",1)
              else:
                  if page_txt != '' or page_txt !=  '':
                      final_text.append([page_txt,pages])
                      originaldoc_data12[page_txt]=pages
                      page_txt = page_txt.replace(page_txt[0:len(page_txt)-1]," ")
      else:
          if page_txt != '' or page_txt !=  '':
              final_text.append([page_txt,pages])
              originaldoc_data12[page_txt]=pages

  return originaldoc_data12

def slicing_original_documents_docx():
    # Convert the docx to a pdf
    convert(filename)
    return slicing_original_documents_pdf(filename[:-4]+"pdf")

def create_output():
  # Exporting CSV File
  import csv
  cols = ["Sentence", "Page Number", "FNN Probability", "CNN Probability", "BERT Probability", "FNN Requirement", "CNN Requirement", "BERT Requirement", "1 out of 3", "2 out of 3", "3 out of 3", "Bert and CNN Agree", "Bert and FNN Agree" ]
  rows = a
  with open('output.csv', 'w', newline='', encoding="utf-8") as f:
      # using csv.writer method from CSV package
      write = csv.writer(f)
      write.writerow(cols[0:13])
      write.writerows(rows)
  
##############################################################################

# Filename of the document to be read.
self.filename = "2010 - fishing.pdf"
if filename[-4:] == "docx":
    sliced_original_doc = slicing_original_documents_docx()
elif filename[-3:] == "pdf":
    sliced_original_doc = slicing_original_documents_pdf(filename)
else:
    print("Please put in .docx or pdf file.")


###### BOW
# Load the BOW model
bow_model = load_model("bow_model.h5")
#Load the BOW tokenizer
bow_tokenizer = load(open('bow_tokenizer.pkl' , 'rb' ))

bow_doc_data = bow_tokenizer.texts_to_matrix(list(sliced_original_doc.keys()))
bow_doc_prediction = bow_model.predict(bow_doc_data)
print("Done Predicting BOW")

######## CNN
# Load the CNN Model
cnn_model = load_model("cnn_model.h5")
max_length = 0 
for l in cnn_model.layers:
    max_length = l.output_shape[1]
    break
cnn_tokenizer = load(open('cnn_tokenizer.pkl' , 'rb' ))
cnn_doc_data = encode_docs(cnn_tokenizer, max_length, list(sliced_original_doc.keys()))
cnn_doc_prediction = cnn_model.predict(cnn_doc_data)
print("Done Predicting CNN")

###### BERT
# Load the BERT Layer for the model.
module_url = 'http://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
bert_tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
bert_model = tf.keras.models.load_model('model_bert.h5',custom_objects={'KerasLayer':hub.KerasLayer})
for l in bert_model.layers:
  bert_max_length = l.output_shape[0][1]
  break
bert_doc_data = bert_encode(list(sliced_original_doc.keys()), bert_tokenizer, max_len = bert_max_length)
bert_doc_prediction = bert_model.predict(bert_doc_data)
print("Done Predicting BERT")

bow_threshold = 0.5
cnn_threshold = 0.5
bert_threshold = 0.5
a = []  
for i in range(len(list(sliced_original_doc.keys()))):
    bow_req = 0
    cnn_req = 0
    bert_req = 0
    oneofthree = 0
    twoofthree = 0
    threeofthree = 0
    bertandcnn = 0
    bertandbow = 0
    if bow_doc_prediction[i][0] >= bow_threshold:
        bow_req = 1
    if cnn_doc_prediction[i][0] >= cnn_threshold:
        cnn_req = 1
    if bert_doc_prediction[i][0] >= bert_threshold:
        bert_req = 1
    if bow_req+cnn_req+bert_req >= 1:
        oneofthree = 1
    if bow_req+cnn_req+bert_req >= 2:
        twoofthree = 1
    if bow_req+cnn_req+bert_req >= 3:
        threeofthree = 1
    if cnn_req+bert_req == 2:
        bertandcnn = 1
    if bow_req+bert_req == 2:
        bertandbow = 1
    
    a.append([list(sliced_original_doc.keys())[i],list(sliced_original_doc.values())[i], bow_doc_prediction[i][0],cnn_doc_prediction[i][0], bert_doc_prediction[i][0], bow_req, cnn_req, bert_req, oneofthree, twoofthree, threeofthree, bertandcnn, bertandbow])

create_output()
  

