
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import SimpleDirectoryReader, VectorStoreIndex, LangchainEmbedding, ServiceContext
import os
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
    
from vectorstore_setup.vectorstore_functions import VectorStoreIndexManager

documents = SimpleDirectoryReader('data/ex_docs').load_data()

#### OpenAI Embeddings
embed_model = OpenAIEmbeddings()

service_context = ServiceContext.from_defaults(embed_model=embed_model)
vector_store_manager = VectorStoreIndexManager(documents, embedding_model=embed_model, service_context=service_context)
vectorstore = vector_store_manager.vectorstore

query_engine = vectorstore.as_query_engine(
    verbose=True, 
)

query_text = "What is something we know about Paul Graham"
response = query_engine.query(query_text)
print(response)

#### Llama model

embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

service_context = ServiceContext.from_defaults(embed_model=embed_model)
vector_store_manager = VectorStoreIndexManager(documents, embedding_model=embed_model, service_context=service_context)
vectorstore = vector_store_manager.vectorstore

query_engine = vectorstore.as_query_engine(
    verbose=True, 
)

query_text = "What is something we know about Paul Graham"
response = query_engine.query(query_text)
print(response)

