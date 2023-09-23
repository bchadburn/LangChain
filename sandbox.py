from langchain.llms import LlamaCpp

from langchain import PromptTemplate, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# Define prompt. Get prompt for specific model from huggingface
template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# See https://python.langchain.com/docs/use_cases/summarization, https://python.langchain.com/docs/integrations/llms/llamacpp

#### Option 1. Stuff (puts all docs into single prompt)
# Define LLM chain

n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="Llama-2-7b-CHAT-GGUF/llama-2-7b-chat.Q2_K.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True, # Verbose is required to pass to the callback manager
)


llm_chain = LLMChain(llm=llm, prompt=prompt)

# Define StuffDocumentsChain
stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain, document_variable_name="text"
)

docs = loader.load()
print(stuff_chain.run(docs))



#