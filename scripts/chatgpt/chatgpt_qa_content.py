from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dotenv

# Reference: https://python.langchain.com/docs/use_cases/question_answering/
# Use case: Have questions about a specific article, website etc and want to avoid adding to internal documents and vectorizing.

dotenv.load_dotenv()

# Load and QA a single document without vectorization
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import WebBaseLoader, PyPDFLoader

chain = load_qa_chain(llm, chain_type="stuff")
# loader = PyPDFLoader("hierarchical_forecasting.pdf")
loader = WebBaseLoader("https://jingwendu.medium.com/predicting-the-future-with-learnings-from-the-m5-competition-19f5841789b9")
docs = loader.load()

max_docs = 6
i = 0
j = max_docs
while i < len(docs):
    tmp_docs = docs[i:j]
    while True:
        question = input("Question: ")
        if question == "exit" or len(question)==0:
            break
        else:
            # Generate a response to the follow-up question
            response = chain({"input_documents": tmp_docs, "question": question}, return_only_outputs=True)['output_text']
            print(response)
   
    # Update the counters
    i += max_docs
    j += max_docs