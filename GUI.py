# This code encompasses GUI used to process document and extract requirements.
# 
# Code Requirements: Utilizes supporting documents
#                  - Slicing Documents.py
#                  - tokenization.py
#                  - Training Code.py
#                  - bow_model.h5, bow_tokenizer.pkl
#                  - cnn_model.h5, cnn_tokenizer.pkl
#                  - model_bert.h5
#                  - tokenization.cpython-38.pyc
#                  - Test-dataset.csv
#                  - Train-val-dataset.csv
#
# Inputs: Document to be processed
# Outputs: Excel document with requirements extracted

# Import packages
import subprocess
import sys
import importlib

packages = ['nltk', 'keras', 'tensorflow', 'tensorflow_hub', 'sentencepiece', 'docx2pdf', 'PyPDF2', 'xlsxwriter', 'PyQt5'] #, 'python-docx'

for package in packages:
    try:
        globals()[package] = importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        globals()[package] = importlib.import_module(package)

try:
    import subprocess
    import sys, os, subprocess, csv
    from PyQt5.QtWidgets import *
    from PyQt5.QtGui import *
    import os.path
    from os import path
    import nltk
    import keras
    import tensorflow as tf
    import tensorflow_hub as hub
    import sentencepiece
    import docx2pdf
    import PyPDF2
    import xlsxwriter
    
# Download missing packages
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'nltk'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'keras'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'tensorflow'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'tensorflow_hub'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'sentencepiece'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'docx2pdf'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'PyPDF2'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'python-docx'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'xlsxwriter'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'PyQt5'])

# Import packages
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
    

import os, csv
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import os.path
from os import path    
import tokenization
import tensorflow as tf
import tensorflow_hub as hub
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

# GUI Class with Models
class App(QWidget):

    # Initialize Window
    def __init__(self):
        super().__init__()
        self.title = 'MITRE Requirements Processing'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480

        # Set User Interface
        self.initUI()
    
    # GUI Window
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create lables
        self.file_label = QLabel(self)
        self.file_label.setText(str("Current selected file: "))
        self.processed_file_data_label = QLabel(self)

        # Create processed file data grid
        self.createGridLayout()

        # Create buttons
        self.createButtonlLayout()

        # Add buttons to GUI
        windowLayout = QVBoxLayout()
        windowLayout.addWidget(self.file_label)     
        windowLayout.addWidget(self.horizontalGrid)
        windowLayout.addStretch()        
        windowLayout.addWidget(self.horizontalGroupBox)

        # Set window layout
        self.setLayout(windowLayout)
        
        # Show GUI
        self.show()
    
    # Creates and organizes buttons
    def createButtonlLayout(self):
        self.filename = ""
        button_layout = QHBoxLayout()
        
        # Select file button
        select_file_button = QPushButton('Select File', self)
        select_file_button.setToolTip('This will open the file explorer to select the file')
        select_file_button.clicked.connect(self.openFileNameDialog)
        button_layout.addWidget(select_file_button)

        # Process Button
        process_button = QPushButton('Process', self)
        process_button.setToolTip('This will process the currently selected file with the AI')
        process_button.clicked.connect(self.process_file)
        button_layout.addWidget(process_button)

        # Open Processed File Button
        open_report_button = QPushButton('Open Processed File Report', self)
        open_report_button.setToolTip('This will open the Excel report generated from the AI')
        open_report_button.clicked.connect(self.open_report)
        button_layout.addWidget(open_report_button)

        # Exit Button
        exit_button = QPushButton('Exit', self)
        exit_button.setToolTip('This will exit the program')
        exit_button.clicked.connect(self.exit_app)
        button_layout.addWidget(exit_button)

        # Create button layout and add to GUI
        self.horizontalGroupBox = QGroupBox("")
        self.horizontalGroupBox.setLayout(button_layout)

    # File explorer function
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "","All Files (*)", options=options)

        # Get filename and update banner
        self.filename = filename
        self.output_file = os.path.splitext(self.filename)[0]+ '_processed.xlsx'
        print(self.output_file)
        self.file_label.setText(str("Selected file: " + self.filename))

    # 'Process' button function
    def process_file(self):
        if self.filename == "":
            self.file_label.setText(str("No file selected.  Please select a file and reprocess."))
            return
        else:
            self.file_label.setText(str("Current processed file: " + self.filename))

        # Convert and process file respectively
        if self.filename[-4:] == "docx":
            self.sliced_original_doc = self.slicing_original_documents_docx()
        elif self.filename[-3:] == "pdf":
            self.sliced_original_doc = self.slicing_original_documents_pdf()
        else:
            print("Please put in .docx or pdf file.")

        # Execute models and generate report
        self.run_bow()
        self.run_cnn()
        self.run_bert()
        self.classify_requirements()
        self.create_output()


    # Function to encode the sentences for the model that uses BERT using a tokenizer
    def bert_encode(self, texts, tokenizer, max_len=512):
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

    # Integer encode and pad documents 
    def encode_docs(self, tokenizer, max_length, train_docs):
        # integer encode 
        encoded = tokenizer.texts_to_sequences(train_docs) 
        # pad sequences 
        padded = pad_sequences(encoded, maxlen=max_length, padding='post') 
        return padded

    # Slice the original document into sentences. Must finally be in PDF format.
    def slicing_original_documents_pdf(self):
        if self.filename[-4:] == "docx":
            pdfobj = open(self.filename[:-4]+"pdf",'rb')
        else:
            pdfobj = open(self.filename[:-4]+".pdf",'rb')
        
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

    # Convert docx to PDF
    def slicing_original_documents_docx(self):
        # Convert the docx to a pdf
        self.docx_filepath = os.path.basename(self.filename)
        convert(self.docx_filepath)
        self.raw_filename = os.path.basename(self.filename)
        originaldoc_data12 = self.slicing_original_documents_pdf()
        return originaldoc_data12

    # Create Excel output
    def create_output(self):
        # Create Excel and tabs
        wb = xlsxwriter.Workbook(self.output_file)
        ws1 = wb.add_worksheet('Summary')
        ws2 = wb.add_worksheet('BOW')
        ws3 = wb.add_worksheet('CNN')
        ws4 = wb.add_worksheet('BERT')
        ws5 = wb.add_worksheet('1 out of 3')
        ws6 = wb.add_worksheet('2 out of 3')
        ws7 = wb.add_worksheet('3 out of 3')
        ws8 = wb.add_worksheet('BERT and CNN')
        ws9 = wb.add_worksheet('BERT and BOW')

        # Excel data
        cols = ["Sentence", "Page Number", "FNN Probability", "CNN Probability", "BERT Probability", "FNN Requirement", "CNN Requirement", "BERT Requirement", "1 out of 3", "2 out of 3", "3 out of 3", "Bert and CNN Agree", "Bert and FNN Agree" ]
        rows = self.a

        # Write column headers
        for col_num, col_val in enumerate(cols):
            ws1.write(0,col_num,col_val)
            ws2.write(0,col_num,col_val)
            ws3.write(0,col_num,col_val)
            ws4.write(0,col_num,col_val)
            ws5.write(0,col_num,col_val)
            ws6.write(0,col_num,col_val)
            ws7.write(0,col_num,col_val)
            ws8.write(0,col_num,col_val)
            ws9.write(0,col_num,col_val)

        # Format cells to highlight requirements
        cell_format = wb.add_format({'bg_color': 'yellow'})
        ws2.conditional_format('$A$1:M%d' % (len(rows)), {'type':   'formula',
                                'criteria': '=INDIRECT("F"&ROW())=1',
                                'format':   cell_format})
        ws3.conditional_format('$A$1:M%d' % (len(rows)), {'type':   'formula',
                                        'criteria': '=INDIRECT("G"&ROW())=1',
                                        'format':   cell_format})
        ws4.conditional_format('$A$1:M%d' % (len(rows)), {'type':   'formula',
                                        'criteria': '=INDIRECT("H"&ROW())=1',
                                        'format':   cell_format})
        ws5.conditional_format('$A$1:M%d' % (len(rows)), {'type':   'formula',
                                        'criteria': '=INDIRECT("I"&ROW())=1',
                                        'format':   cell_format})
        ws6.conditional_format('$A$1:M%d' % (len(rows)), {'type':   'formula',
                                        'criteria': '=INDIRECT("J"&ROW())=1',
                                        'format':   cell_format})
        ws7.conditional_format('$A$1:M%d' % (len(rows)), {'type':   'formula',
                                        'criteria': '=INDIRECT("K"&ROW())=1',
                                        'format':   cell_format})
        ws8.conditional_format('$A$1:M%d' % (len(rows)), {'type':   'formula',
                                        'criteria': '=INDIRECT("L"&ROW())=1',
                                        'format':   cell_format})
        ws9.conditional_format('$A$1:M%d' % (len(rows)), {'type':   'formula',
                                        'criteria': '=INDIRECT("M"&ROW())=1',
                                        'format':   cell_format})
 

        # Write data into excel
        num_of_bow_req = 0
        num_of_cnn_req = 0
        num_of_bert_req = 0
        num_of_one_three_req = 0
        num_of_two_three_req = 0
        num_of_three_three_req = 0
        num_of_bert_cnn_req = 0
        num_of_bert_bow_req = 0

        for row_num, row in enumerate(rows):
            for col_num, col_val in enumerate(row):
                ws1.write(row_num+1,col_num,col_val)
                ws2.write(row_num+1,col_num,col_val)
                ws3.write(row_num+1,col_num,col_val)
                ws4.write(row_num+1,col_num,col_val)
                ws5.write(row_num+1,col_num,col_val)
                ws6.write(row_num+1,col_num,col_val)
                ws7.write(row_num+1,col_num,col_val)
                ws8.write(row_num+1,col_num,col_val)
                ws9.write(row_num+1,col_num,col_val)

                if col_num == 5 and col_val == 1:
                    num_of_bow_req = num_of_bow_req + 1
                elif col_num == 6 and col_val == 1:
                    num_of_cnn_req = num_of_cnn_req + 1
                elif col_num == 7 and col_val == 1:
                    num_of_bert_req = num_of_bert_req + 1
                elif col_num == 8 and col_val == 1:
                    num_of_one_three_req = num_of_one_three_req + 1
                elif col_num == 9 and col_val == 1:
                    num_of_two_three_req = num_of_two_three_req + 1
                elif col_num == 10 and col_val == 1:
                    num_of_three_three_req = num_of_three_three_req + 1
                elif col_num == 11 and col_val == 1:
                    num_of_bert_cnn_req = num_of_bert_cnn_req + 1
                elif col_num == 12 and col_val == 1:
                    num_of_bert_bow_req = num_of_bert_bow_req + 1
             
        wb.close()

        # Update GUI
        self.layout.addWidget(QLabel(str(len(rows)) + ' sentences found in document'),0,0)
        self.layout.addWidget(QLabel(str(num_of_bow_req) + ' requirements identified using BOW'),1,0)
        self.layout.addWidget(QLabel(str(num_of_cnn_req) + ' requirements identified using CNN'),2,0)
        self.layout.addWidget(QLabel(str(num_of_bert_req) + ' requirements identified using BERT'),3,0)
        self.layout.addWidget(QLabel(str(num_of_one_three_req) + ' requirements identified using where 1 of 3 models identify requirement'),4,0)
        self.layout.addWidget(QLabel(str(num_of_two_three_req) + ' requirements identified using where 2 of 3 models identify requirement'),5,0)
        self.layout.addWidget(QLabel(str(num_of_three_three_req) + ' requirements identified using where 3 of 3 models identify requirement'),6,0)
        self.layout.addWidget(QLabel(str(num_of_bert_cnn_req) + ' requirements identified where BERT and CNN agree'),7,0)
        self.layout.addWidget(QLabel(str(num_of_bert_bow_req) + ' requirements identified where BERT and BOW agree'),8,0)

        # Update grid
        self.update_grid()

    # BOW Model
    def run_bow(self):
        # Load the BOW model
        bow_model = load_model("bow_model.h5")
        #Load the BOW tokenizer
        bow_tokenizer = load(open('bow_tokenizer.pkl' , 'rb' ))

        bow_doc_data = bow_tokenizer.texts_to_matrix(list(self.sliced_original_doc.keys()))
        self.bow_doc_prediction = bow_model.predict(bow_doc_data)
        print('Done Predicting BOW')
        return

    # CNN Model
    def run_cnn(self):
        # Load the CNN Model
        cnn_model = load_model("cnn_model.h5")
        max_length = 0 
        for l in cnn_model.layers:
            max_length = l.output_shape[1]
            break
        cnn_tokenizer = load(open('cnn_tokenizer.pkl' , 'rb' ))
        cnn_doc_data = self.encode_docs(cnn_tokenizer, max_length, list(self.sliced_original_doc.keys()))
        self.cnn_doc_prediction = cnn_model.predict(cnn_doc_data)
        print('Done Predicting CNN')
        return
    
    # BERT Model
    def run_bert(self):
        # Load the BERT Layer for the model.
        module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'
        bert_layer = hub.KerasLayer(module_url, trainable=True)

        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        bert_tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
        bert_model = tf.keras.models.load_model('model_bert.h5',custom_objects={'KerasLayer':hub.KerasLayer})
        for l in bert_model.layers:
            bert_max_length = l.output_shape[0][1]
            break
        bert_doc_data = self.bert_encode(list(self.sliced_original_doc.keys()), bert_tokenizer, max_len = bert_max_length)
        self.bert_doc_prediction = bert_model.predict(bert_doc_data)
        print('Done Predicting BERT')
        return

    # Classify requirements and organize output data 
    def classify_requirements(self):
        bow_threshold = 0.5
        cnn_threshold = 0.5
        bert_threshold = 0.5
        self.a = []  
        for i in range(len(list(self.sliced_original_doc.keys()))):
            bow_req = 0
            cnn_req = 0
            bert_req = 0
            oneofthree = 0
            twoofthree = 0
            threeofthree = 0
            bertandcnn = 0
            bertandbow = 0
            if self.bow_doc_prediction[i][0] >= bow_threshold:
                bow_req = 1
            if self.cnn_doc_prediction[i][0] >= cnn_threshold:
                cnn_req = 1
            if self.bert_doc_prediction[i][0] >= bert_threshold:
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
            
            self.a.append([list(self.sliced_original_doc.keys())[i],list(self.sliced_original_doc.values())[i], self.bow_doc_prediction[i][0], self.cnn_doc_prediction[i][0], self.bert_doc_prediction[i][0], bow_req, cnn_req, bert_req, oneofthree, twoofthree, threeofthree, bertandcnn, bertandbow])

    # Grid layout for GUI information
    def createGridLayout(self):
        self.horizontalGrid = QGroupBox('')
        self.layout = QGridLayout()

    # Update GUI
    def update_grid(self):  
        self.horizontalGrid.setLayout(self.layout)

    # 'Open Report' button function
    def open_report(self):
        if os.path.isfile(self.output_file):
            cmd = 'open ' + '\"' + self.output_file + '\"'
            os.system(cmd)
        else:
            self.file_label.setText(str('Output file does not exist for selected folder. Please process before opening output file.'))
            return

    # Exit application
    def exit_app(self):
        QApplication.quit()

# Run application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())