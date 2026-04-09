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

# ====== CONFIG ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ====== FUNCTIONS ======

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
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_texts(chunks, embedding=embeddings)

def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0,model="gpt-4o-mini")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ====== STREAMLIT APP ======

st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
st.title("Chat with your PDF (RAG)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading and indexing PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        chunks = split_text(raw_text)
        vectorstore = create_vectorstore(chunks)
        qa_chain = create_qa_chain(vectorstore)
    st.success("PDF loaded and ready!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Ask a question about the PDF")

    if query:
        with st.spinner("Getting answer..."):
            result = qa_chain.run(query)
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot", result))

    for speaker, msg in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {msg}")