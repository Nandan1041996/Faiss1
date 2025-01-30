import time
import os
import gc
import json
import pickle
import secrets
import pandas as pd
import pandas as pd
from dotenv import load_dotenv
from functions import get_chain,get_answer
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import CSVLoader,TextLoader,PyPDFLoader,UnstructuredExcelLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from flask import Flask,request, render_template, redirect, url_for, flash, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter


from sentence_transformers import SentenceTransformer

# Load a pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can replace this with another model

df = pd.read_excel(r'C:\Users\G01889\OneDrive\Documents\faiss\Quest_Ans_For_All_Ext-txt-pdf.csv-1xlsx-1\Document\anc.xlsx')

que_lst = df['Question'].to_list()

# print(len(que_lst))
# Convert text to embeddings
embeddings = model.encode(que_lst)


str_embed = []
for i in embeddings:
    str_embed.append(str(i.tolist()))


df['embedding'] = str_embed



print(df)
print(df.columns)


df.to_csv('embed1.csv',index=False)

 




