from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dotenv

# Reference: https://python.langchain.com/docs/use_cases/question_answering/
# Use case: Have questions about your own text documents 

# The pipeline for converting raw unstructured data into a QA chain looks like this:

# Loading: First we need to load our data. Unstructured data can be loaded from many sources. Use the LangChain integration hub to browse the full set of loaders. Each loader returns data as a LangChain Document.
# Splitting: Text splitters break Documents into splits of specified size
# Storage: Storage (e.g., often a vectorstore) will house and often embed the splits
# Retrieval: The app retrieves splits from storage (e.g., often with similar embeddings to the input question)
# Generation: An LLM produces an answer using a prompt that includes the question and the retrieved data
# Conversation (Extension): Hold a multi-turn conversation by adding Memory to your QA chain.

dotenv.load_dotenv()

#### Step 1. Load

# URL is our example blog post to create QA for
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/") # See other langchain loaders in ReadMe.md
data = loader.load()

#### Step 2. Split
# DocumentSplitters are just one type of the more generic DocumentTransformers, which can all be useful in this preprocessing step.
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)

#### Step 3. Store

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

#### Step 4. Retrieve
# Vectorstores are commonly used for retrieval, but they are not the only option. For example, SVMs (see thread here) can also be used.
question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
len(docs)

# SVM example of retrieval

from langchain.retrievers import SVMRetriever

svm_retriever = SVMRetriever.from_documents(all_splits, OpenAIEmbeddings())
docs_svm=svm_retriever.get_relevant_documents(question)
len(docs_svm)

# Some common ways to improve on vector similarity search include:

# MultiQueryRetriever generates variants of the input question to improve retrieval.
# Max marginal relevance selects for relevance and diversity among the retrieved documents.
# Documents can be filtered during retrieval using metadata filters.

import logging

from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(),
                                                  llm=ChatOpenAI(temperature=0))
unique_docs = retriever_from_llm.get_relevant_documents(query=question)
len(unique_docs)

from langchain.chains import RetrievalQA

#### Step 5. Generate
# Distill the retrieved documents into an answer using an LLM/Chat model (e.g., gpt-3.5-turbo) with RetrievalQA chain.
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
qa_chain({"query": question})

# Note, you can pass in an LLM or a ChatModel (like we did here) to the RetrievalQA chain.

# Return source documents
# The full set of retrieved documents used for answer distillation can be returned using return_source_documents=True.

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever(),
                                       return_source_documents=True)
result = qa_chain({"query": question})
print(len(result['source_documents']))
result['source_documents'][0]

# Customizing retrieved document processing
# Retrieved documents can be fed to an LLM for answer distillation in a few different ways such as stuff, refine, map-reduce, and map-rerank
# stuff is commonly used because it simply "stuffs" all retrieved documents into the prompt.
# The load_qa_chain is an easy way to pass documents to an LLM using these various approaches (e.g., see chain_type).

from langchain.chains.question_answering import load_qa_chain

refernce_docs = result['source_documents'] # Source documents
chain = load_qa_chain(llm, chain_type="stuff")
chain({"input_documents": refernce_docs, "question": question},return_only_outputs=True)

# Another option is to pass the chain_type to RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever(),
                                       chain_type="stuff")
result = qa_chain({"query": question})
print("query:", result['query'])
print("Response:", result['result'])