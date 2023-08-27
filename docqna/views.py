from django.shortcuts import render
from chat.views import llm, tokenizer
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from django.http import JsonResponse

embeddings = HuggingFaceEmbeddings()

persist_directory = "db"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#Define the chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl = 'duckdb+parquet',
    persist_directory = "db",
    anonymized_telemetry = False
)
# Initializing chroma client
persistent_client = chromadb.Client(settings=CHROMA_SETTINGS)

_LANGCHAIN_DEFAULT_COLLECTION_NAME="langchain-collection"

langchain_chroma = Chroma(
    client=persistent_client,
    collection_name=_LANGCHAIN_DEFAULT_COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=persist_directory,
    client_settings=CHROMA_SETTINGS
)

langchain_chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory, collection_name=name)

langchain_chroma.as_retriever(k=2)

def loadDocument():
    loader = PyPDFLoader("/content/back_pain_treatment.pdf")
    pages = loader.load()
    return pages

def splitText(pages):
    chunks = text_splitter.split_documents(pages)
    return chunks



def get_collection_list():
    return persistent_client.list_collections()

def create_collection(name, embeddings):
    return persistent_client.create_collection(name=name, embedding_function=embeddings)

def see_collection_data(name):
    collection = persistent_client.get_collection(name=name)
    return collection.peek()


def get_document(request):
    if request.method=='POST':
        file_uploaded = request.FILES.get('pdf_file')
        return JsonResponse("something")