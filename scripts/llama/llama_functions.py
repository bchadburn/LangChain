from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline


class CreatePipeline:
    def __init__(
        self,
        model_path,
        device_map="auto",
        max_length=800,
        top_p=0.95,
        repetition_penalty=1.15,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
        )
        pipe = pipeline(
            "text-generation",
            model=self.model,
            do_sample=True,
            tokenizer=self.tokenizer,
            max_length=max_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
