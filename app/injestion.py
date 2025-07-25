import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils import parse_file

def ingest_document(file_path: str, user_id: str):

    documents = parse_file(file_path) #parse file into raw text

   
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) 
    chunks = splitter.split_documents(documents) #split files into chunks

    
    Qdrant.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        url="http://localhost:6333",
        collection_name=f"user_{user_id}_docs" 
    ) #store in related user quadrants

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python injestion.py <file_path> <user_id>")
    else:
        ingest_document(sys.argv[1], sys.argv[2])
