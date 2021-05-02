import docx2pdf
import PyPDF2
import string
import re 
import csv
import pandas as pd
from docx2pdf import convert


def slicing_original_documents_docx(filename):
    # Convert the docx to a pdf
    convert(filename)
    return slicing_original_documents_pdf(filename[:-4]+"pdf")


def slicing_original_documents_pdf(filename):
  pdfobj = open(filename,'rb')
  pdfreader = PyPDF2.PdfFileReader(pdfobj)
  page_counts = pdfreader.numPages
  # pageobj = pdfreader.getPage(16)
  final_text = []
  for i in range(page_counts):
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
                      final_text.append(page_txt[corpus_slice])
                      page_txt = page_txt.replace(page_txt[0:period_idx+1]," ")
                  elif page_txt[period_idx-1] == ")" and (page_txt[period_idx+1].isspace() or page_txt[period_idx+1].isalpha()):
                      corpus_slice = slice(period_idx+1)
                      final_text.append(page_txt[corpus_slice])
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
                  final_text.append(page_txt)
                  page_txt = page_txt.replace(page_txt[0:len(page_txt)-1]," ")
      else:
          final_text.append(page_txt)
  final_text = [i for i in final_text if i != '' or i !=  '']
  ready_to_exce = pd.DataFrame(final_text)
  ready_to_exce.to_csv("Sliced_Original_Document.csv",index=False,header=False)

def creating_raw_list():
  originaldoc_data = []
  with open("Test-dataset_copy.csv",'r',encoding = "utf8",errors = "ignore") as data: #in final model: 1.csv
      reader = csv.reader(data)
      originaldoc_data = list(reader)
  originaldoc_data1 = []
  for i in originaldoc_data:
    for j in i:
      originaldoc_data1.append(j)
  #originaldoc_data12 = np.array(originaldoc_data1)
  originaldoc_data12 = originaldoc_data1
  return originaldoc_data12
########################################################################################################################  
 

# Put file name to read here 
filename = "2007 - nlm.docx"
if filename[-4:] == "docx":
    sliced_original_doc = slicing_original_documents_docx(filename=filename)
elif filename[-3:] == "pdf":
    sliced_original_doc = slicing_original_documents_pdf(filename)
else:
    print("Please put in .docx or pdf file.")