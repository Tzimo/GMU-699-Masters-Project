# Import packages
import os, datetime, nltk, docx, pptx, PyPDF2, sys, subprocess, tkinter as tk

from tkinter.filedialog import askopenfilename


# Datasets

raw_sentences = []

def read_file(filename):
    doc = docx.Document(filename)
    for line in doc.paragraphs:
        #print(line.text)
        raw_sentences.append(line.text)


def main():

    filename = 'C:\\Users\\Michael\\Documents\\GMU\\SYST 699 Masters Project\\GMU-699-Masters-Project\\Tzimourakas_Paper_Review1.docx'
    #if len(sys.argv) !=2:
        # Open gui to select file
    #    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    #else:
        # Get file
    #    filename = str(sys.argv[1])
    
    print(filename)
    read_file(filename)


    processed_sentences = list(filter(None, raw_sentences))
    for sentance in processed_sentences:
        print(sentance)


if __name__ == '__main__':
    main()



