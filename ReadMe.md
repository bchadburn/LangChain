
To convert existing GGML models to GGUF you can run the following in llama.cpp (https://python.langchain.com/docs/integrations/llms/llamacpp):
python ./convert-llama-ggmlv3-to-gguf.py --eps 1e-5 --input models/openorca-platypus2-13b.ggmlv3.q4_0.bin --output models/openorca-platypus2-13b.gguf.q4_0.bin
