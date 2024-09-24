import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## Få en direkt insikt om dina dokument
## Hur det fungerar
Vänligen följ instruktionerna
1.) Skriv in din API-nyckel. Den går att hämta från aistudio.google.com.
2) ladda upp dina dokument. 
3) ställ en fråga på engelska. 


 """)

api_key = st.text_input("skriv in din API nyckel:", type="password", key="api_key_input")

def get_pdf_text(pdf_docs):
   text=""
   for pdf in pdf_docs:
       pdf_reader= PdfReader(pdf)
       for page in pdf_reader.pages:
           text += page.extract_text()
       return text     
def get_text_chunks(text):
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
   chunks = text_splitter.split_text(text)
   return chunks

def get_vector_store(text_chunks, api_key):
   embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
   vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
   vector_store.save_local("faiss_index")


def get_conversational_chain(api_key):
    prompt_template = """Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not available in the document, don't provide the wrong answer.
    context: \n {context}?\n
    Question: \n{question} \n

    Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    
    # Correct usage of PromptTemplate
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    # Pass api_key to get_conversational_chain
    chain = get_conversational_chain(api_key)
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply", response["output_text"])


#streamlit ui

def main(): 
    st.header("socument geneistreamlit_app.py")

    user_question = st.text_input("ask a question from PDF files", key="user_question")
    if user_question and api_key: 
        user_input(user_question, api_key)
    
    with st.sidebar: 
        st.title("menu:")
        pdf_docs = st.file_uploader("upload your docs", accept_multiple_files=True, key="pdf_uploader")
        if st.button("submit & process", key="process_button") and api_key:
            with st.spinner("processing..."): 
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("done")

if __name__ == "__main__": 
    main()



    

