
import os
from dotenv import load_dotenv
from glob import glob
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = PineconeVectorStore(
    index_name="vectortest", embedding=embeddings_model
)
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
llm = ChatOpenAI()
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

print(retrieval_chain.invoke({"input": "What is the most GK about?"})["answer"])