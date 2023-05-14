from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI

def load_env():
    load_dotenv()
    st.set_page_config(page_title="ChatPDF")
    st.header("ChatPDF")

def upload_file():
    pdf = st.file_uploader("Upload PDF", type="pdf")
    return pdf

def extract_text(pdf):
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    return None

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

def get_user_query():
    query = st.text_input("Ask a question about PDF:", value="", key='1')
    return query

def get_response(docs, user_query):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_query)
        print(cb)
        return response

def main():
    load_env()
    pdf = upload_file()
    text = extract_text(pdf)
    if text:
        chunks = split_text(text)
        knowledge_base = create_embeddings(chunks)
        query = get_user_query()
        if query:
            relevant_docs = knowledge_base.similarity_search(query)
            response = get_response(relevant_docs, query)
            st.write(response)

if __name__ == '__main__':
    main()
