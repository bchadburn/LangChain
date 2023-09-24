from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import LlamaCpp
import json
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.helper_functions import parse_agent_response

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

llm = LlamaCpp(
    model_path="Llama-2-7b-CHAT-GGUF/llama-2-7b-chat.Q2_K.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True, # Verbose is required to pass to the callback manager
)

from langchain.llms import OpenAI
llm = OpenAI(temperature=0)

toolkit = load_tools(["serpapi"], llm=llm)

agent = initialize_agent(toolkit, llm, agent="zero-shot-react-description", verbose=True, return_intermediate_steps=True)

response = agent({"input":"what was the first album of the" 
                    "band that Natalie Bergman is a part of?"})


parse_agent_response(response)

print(json_data)