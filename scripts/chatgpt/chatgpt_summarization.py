from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.chains.summarize import load_summarize_chain

# See https://python.langchain.com/docs/use_cases/summarization

model_version = "gpt-3.5-turbo"
# model_version = "gpt-4"
llm = ChatOpenAI(temperature=0, model_name=model_version)

chain = load_summarize_chain(llm, chain_type="stuff")
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

print(chain.run(docs))
    

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# Define prompt
prompt_template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

# Define LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Define StuffDocumentsChain
stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain, document_variable_name="text"
)

docs = loader.load()
print(stuff_chain.run(docs))