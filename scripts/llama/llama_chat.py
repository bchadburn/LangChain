from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

template = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
{prompt}
"""

prompt_template = PromptTemplate(template=template, input_variables=["prompt"])

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


llm_chain = LLMChain(prompt=prompt_template, llm=llm)
prompt = "What NFL team won the Super Bowl in the year Justin Bieber was born?"
llm_chain.run(prompt)  # Can also use llm(question) instead of LLM chain


# Converstaion history
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a chatbot having a conversation with a human."), # The persistent system prompt
    MessagesPlaceholder(variable_name="chat_history"), # Where the memory will be stored.
    HumanMessagePromptTemplate.from_template("{human_input}"), # Where the human input will injected
])
    
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)

llm_chain.predict(human_input="Hi there my friend")

llm_chain.predict(human_input="Not too bad - how are you?")

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

conversation = ConversationChain(
    llm=llm, verbose=False, memory=ConversationBufferMemory(ai_prefix="AI Assistant")
)

conversation.predict(input="Hi there!")

conversation.predict(input="What's the weather?")

# Not sure how this is set up with LLAMA models yet. Probably need to parse history.messages to feed in the right format
# from langchain.memory import ChatMessageHistory

# history = ChatMessageHistory()

# history.add_ai_message("hi!")

# history.add_user_message("what is the capital of france?")

# history.messages

# ai_response = llm(history.messages)

# print(ai_response.content)

# history.add_ai_message(ai_response.content)
# history.messages
