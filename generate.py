# SIU KING WAI SM4701 Deepstory
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class Generator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(f'data/gpt2/{model_name}')
        self.model = GPT2LMHeadModel.from_pretrained(f'data/gpt2/{model_name}').to('cuda')

    def generate(self, text, max_length, top_p, top_k, temperature, do_sample):
        # encode input context
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to('cuda') if text else None
        outputs = self.model.generate(input_ids=input_ids,
                                      max_length=max_length,
                                      top_p=top_p,
                                      top_k=top_k,
                                      temperature=temperature,
                                      do_sample=do_sample)
        return self.tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True)
