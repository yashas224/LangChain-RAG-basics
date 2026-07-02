import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

load_dotenv()


if __name__ == "__main__":
    print("loading Documnts")
    cwd = os.getcwd()
    loader = TextLoader(file_path=cwd + "/medium-blog.txt")
    documentList: list[Document] = loader.load()

    print("Splitting into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splitDocuments: list[Document] = text_splitter.split_documents(documentList)
    print(f"created {len(splitDocuments)} chunks")

    print("Ceating Embeddings")

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    print("Starting to Ingest into Vector store")
    index = os.getenv("VECTOR_STORE_INDEX_NAME")
    vector_store = PineconeVectorStore(index_name=index, embedding=embeddings)
    vector_store.add_documents(splitDocuments)
    print("Finished Ingesting into Vector store")
