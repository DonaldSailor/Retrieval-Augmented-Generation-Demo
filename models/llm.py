from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

class GenerativeModel:

    def __init__(self, model_name:str) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device=0
        )

    
    def generate_answer(self, query:str, retrieved_document:str, max_length:int=256) -> str:
        sequence = self.pipeline(
        f"Context: '''{retrieved_document}''' Question: '''{query}''' \
            answer the Question according to the Context.",
        max_length=max_length,
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=self.tokenizer.eos_token_id,
        )

        return f"Result: {sequence[0]['generated_text']}"