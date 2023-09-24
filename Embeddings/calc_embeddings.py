from langchain.embeddings import OpenAIEmbeddings
import os
import sys
from llama_index import SimpleDirectoryReader, VectorStoreIndex, LangchainEmbedding, ServiceContext


#### OpenAI Embeddings
embed_model = OpenAIEmbeddings() # Specify OPENAI_API_KEY in .env file
service_context = ServiceContext.from_defaults(embed_model=embed_model)

documents = SimpleDirectoryReader('data/ex_docs').load_data()
new_index = VectorStoreIndex.from_documents(
    documents, 
    service_context=service_context,
)

query_engine = new_index.as_query_engine(
    verbose=True, 
)


query_text = "Who is Paul Graham"
response = query_engine.query(query_text)
print(response)

#### Llama model
from langchain.embeddings import LlamaCppEmbeddings

embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

service_context = ServiceContext.from_defaults(embed_model=embed_model)
vectorstore = VectorStoreIndex.from_documents(
    documents, 
    service_context=service_context,
)

query_engine = new_index.as_query_engine(
    verbose=True, 
)

query_text = "Who is Paul Graham"
response = query_engine.query(query_text)
print(response)

