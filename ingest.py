#import libraries
import os

import langchain
import pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

#Lets Read the document 
def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents   = file_loader.load()
    return documents

doc = read_doc('documents/')
print(doc)

#Divide the docs into chunks
def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return docs

documents = chunk_data(docs=doc)
print(documents)

#Embedding Technique of OPENAI
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=os.environ["GEMINI_API_KEY"])
print(embeddings)

vectors = embeddings.embed_query("How are you?")
print(vectors)