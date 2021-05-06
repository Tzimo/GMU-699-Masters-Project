# GMU-699-Masters-Project
You must download GUI.py and all the supporting files in order to run this program.

GUI.py: Main program to run GUI and process file for prediction.

Deliverable 1: Sentence Slicer 

 

For using the Document Slicer software package, the user should take the following steps: 

 

The format of the document should be either docx or pdf. 

The python code and the document files should be saved in the same directory.  

The user should copy and paste the name of the file, with its extension, in the line 80 of the code within the quote.  

The software package is ready to run. 

 

Deliverable 2: Model Trainer 

 

The python code, called “training code.py”, and the training data set in the csv format should be in the same directory. This csv file contains all of the labeled data. One column has the sentences and the second column are the 0,1 labels of where the sentence is a requirement. 

The user should copy and paste the name of the dataset file, with its extension (.csv), in the line 235 of the code within the quote. The name of the file currently is “Train-val-dataset.csv”. 

If training the BERT model, the tokenizer file also needs to be in the same file, called “tokenization.py”. 

After running the code, 5 files will be generated and saved in the same folder. These include the tokenizer file for the CNN model and the FNN (BOW) model and the trained model of all three models.  

The last three lines are the lines of code in the file are used to start the training process of the three models. Initially, they are all set to run. However, if any of them should not be run and trained, then they can be commented out and the specific model will not be trained.  

If the user would like to modify the hyperparameters of the neural networks, they can be modified in the definition functions of the models. They are called define_BOW_model(), define_CNN_model(), and for the BERT model, build_model(). More description of the models are in the final report of the project. 

 

Deliverable 3: Requirement Predictor (GUI) 

 

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
