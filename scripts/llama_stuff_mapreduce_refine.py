from langchain.llms import LlamaCpp
from langchain.document_loaders import WebBaseLoader
from langchain.callbacks.manager import CallbackManager
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Define prompt
template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:"""

prompt = PromptTemplate(template=template, input_variables=["text"])

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

# Set up LLM - GPU Supported
model_path = "Llama-2-7b-CHAT-GGUF/llama-2-7b-chat.Q2_K.gguf"
n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Define LLM chain
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx = 512,
    callback_manager=callback_manager,
    verbose=True, # Verbose is required to pass to the callback manager
)

# Use load_summarize_chain with chain_type="stuff", we will use the StuffDocumentsChain.

# Run if n_ctx is 4000+. Else, use other solutions to reduce sequence size.
# chain = load_summarize_chain(llm, chain_type="stuff")
# chain.run(docs)

#### Option 1. Stuff - Uses StuffDocumentsChain directly (Returns same result as using load_summarize_chain(llm, chain_type="stuff") above
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# We can also supply chain_type="map_reduce" or chain_type="refine" (read more https://python.langchain.com/docs/modules/chains/document/refine).
llm_chain = LLMChain(llm=llm, prompt=prompt)
chain = load_summarize_chain(llm, chain_type="stuff")

# Define StuffDocumentsChain
stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain, document_variable_name="text"
)

# Run Directly if n_ctx is large enough
# docs = loader.load()
# print(stuff_chain.run(docs))
##### Option 2. Map-Reduce: Summarize each document on it's own in a "map" step and then "reduce" the summaries into a final summary (see here for more on the MapReduceDocumentsChain, which is used for this method).

from langchain.chains.mapreduce import MapReduceChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain

# Map
map_template = """The following is a set of documents
{docs}
Based on this list of docs, please identify the main themes 
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

"""The ReduceDocumentsChain handles taking the document mapping results and reducing them into a single output. It wraps a generic CombineDocumentsChain (like StuffDocumentsChain) but 
adds the ability to collapse documents before passing it to the CombineDocumentsChain if their cumulative size exceeds token_max. In this example, we can actually re-use our chain for 
combining our docs to also collapse our docs."""

# Reduce
reduce_template = """The following is set of summaries:
{doc_summaries}
Take these and distill it into a final, consolidated summary of the main themes. 
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)

# Run chain
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="doc_summaries"
)

# Combines and iteravely reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=500,
)

max_token_count = 500

# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=max_token_count, chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)
# print(map_reduce_chain.run(split_docs))


##### Option 3. Refine

# The refine documents chain constructs a response by looping over the input documents and iteratively updating its answer. For each document, it passes all non-document inputs, the current document, and the latest intermediate answer to an LLM chain to get a new answer.

chain = load_summarize_chain(llm, chain_type="refine")
chain.run(split_docs)


# It's also possible to supply a prompt and return intermediate steps.

refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary in Italian"
    "If the context isn't useful, return the original summary."
)
refine_prompt = PromptTemplate.from_template(refine_template)
chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=prompt,
    refine_prompt=refine_prompt,
    return_intermediate_steps=True,
    input_key="input_documents",
    output_key="output_text",
)

# Only running a couple of the splits as some exceed max_token size
result = chain({"input_documents": split_docs[5:7]}, return_only_outputs=True)

print(result["output_text"])
print("\n\n".join(result["intermediate_steps"][:3]))