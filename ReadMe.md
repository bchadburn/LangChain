# LLM LangChain Repository

This repository contains scripts and resources related to the LLM (Language Model) and LangChain for various natural language processing tasks.

## Project Structure

### chroma_db

The `chroma_db` folder contains scripts and utilities related to Chroma, a component of LangChain for vector representation.

### vectorstore_setup

The `vectorstore_setup` folder includes scripts for setting up and configuring vector stores for LangChain.

### utils

The `utils` folder houses general-purpose and helper functions used across the project.

### scripts

The `scripts` folder consists of specific scripts that can be utilized for various LLM tasks including chat, question-answering, summarization, embeddings, and agent usage. 

### Notebooks

The `Notebooks` directory contains Jupyter notebooks that provide walkthroughs, tutorials, and examples for performing LLM tasks using LangChain. 

### Models

All pre-trained models should be placed directly below the project level directory. For GGUF models with multiple model files, include the parent folder at the project level directory. For example, if you have a model named "Llama-2-7b-Chat-GGUF," you should create a folder with that name and include the model files, such as `gguf_model.pth` and `config.json`, within it.

### Data

The `Data` folder is where you store data used for indexing or configuring vector stores. 

## API Keys

For secure and organized handling of API keys, we recommend placing them in a `.env` file at the project's root level. This allows you to access the API keys without exposing them directly in your code. For example, if you use the OpenAI API, you can store your OPENAI_API_KEY in the `.env` file, and your scripts can access it from there.


## Additional resources

### Load
See all langChain loaders here: https://integrations.langchain.com/

### Store
Storing embeddings for PDF for later retriveal of similar docs
https://github.com/mayooear/gpt4-pdf-chatbot-langchain

### Models 
To convert existing GGML models to GGUF you can run the following in llama.cpp (https://python.langchain.com/docs/integrations/llms/llamacpp):
python ./convert-llama-ggmlv3-to-gguf.py --eps 1e-5 --input models/openorca-platypus2-13b.ggmlv3.q4_0.bin --output models/openorca-platypus2-13b.gguf.q4_0.bin
