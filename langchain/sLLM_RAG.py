# pip install -q python-dotenv
# pip install -q langchain chromadb 
# pip install -q streamlit langchain chromadb python-dotenv
# pip install -q torch sentence_transformers huggingface_hub pypdf

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import chroma
from langchain.embeddings import Senten