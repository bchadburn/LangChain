from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import os
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_utils.config import MODEL_PATH
from llm_utils.llm_functions import CreatePipeline

template = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
{prompt}
"""

prompt_template = PromptTemplate(template=template, input_variables=["prompt"])

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Load GPTQ model
# https://integrations.langchain.com/llms doesn't list TheBloke GPTQ model. So we can use HuggingFacePipeline wrapper to ensure compatibility with LangChain.
pipeline = CreatePipeline(
    MODEL_PATH, device_map="auto", max_length=256, top_p=0.95, repetition_penalty=1.25
)
llm = pipeline.llm

# n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
# n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# llm = LlamaCpp(
#     model_path="text-generation-webui/models/TheBloke_Phind-CodeLlama-34B-v2-GPTQ/",
#     n_gpu_layers=n_gpu_layers,
#     n_batch=n_batch,
#     callback_manager=callback_manager,
#     verbose=True,  # Verbose is required to pass to the callback manager
# )

llm_chain = LLMChain(prompt=prompt_template, llm=llm)

f = open("LangChain/data/chats/queries.txt", "r")
content = f.readlines()

# Return answer from first question
prompt = content[0]
llm_chain.run(prompt)  # Can also use llm(question) instead of LLM chain
