

import os
from dotenv import load_dotenv
from glob import glob
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_community.document_loaders.csv_loader import CSVLoader

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


# 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
directory = "./data/"

#Load your CSV files

csv_files = glob(os.path.join(directory, '*.csv'))

# Initialize an empty list to store data from all CSV files
all_data = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)

# Iterate through each CSV file and load its data
for file_path in csv_files:
    loader = CSVLoader(file_path=file_path, encoding="utf-8",)
    data = loader.load()
    all_data.append(data)

for csvdata in all_data:
    chunks = text_splitter.split_documents(csvdata)
    docsearchcsv = PineconeVectorStore.from_documents(chunks, embeddings_model, index_name="vectortest")


# Load your PDF files
PDF_loader = PyPDFDirectoryLoader(directory)
pdf = PDF_loader.load()
chunks = text_splitter.split_documents(pdf)

docsearchpdf = PineconeVectorStore.from_documents(chunks, embeddings_model, index_name="vectortest")

print(docsearchpdf)