import streamlit as st
from PyPDF2 import PdfReader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

#config api key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#functions
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def create_vectorstore(chunks):
    embeddings =OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_texts(chunks, embeeding=embeddings)

def create_qa_chain(vectorstore):
    retriver = vectorstore.as_retriver()
    llm = ChatOpenAI (openai_api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o-mini")

st.title ("Rag Chatbot")