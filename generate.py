# SIU KING WAI SM4701 Deepstory
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class Generator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(f'data/gpt2/{model_name}')
        self.model = GPT2LMHeadModel.from_pretrained(f'data/gpt2/{model_name}').to('cuda')
        with open(f'data/gpt2/{model_name}/default.txt', 'r') as f:
            text = f.read()
        if text[-1] == '\n':
            text = text[:-1]
        self.default = text

    def generate(self, text, predict_length, top_p, top_k, temperature, do_sample, num=1):

        if text:
            # encode input context to gpt2 tokens
            input_ids = self.tokenizer.encode(text, return_tensors='pt').to('cuda')
            # gpt2 model can only infer to maximum of 1024 tokens
            if len(input_ids[0]) + predict_length > 1024:
                # take the nearest (1024 - predict_length) tokens from the end while reserving space to generate.
                input_ids = input_ids[0][-(1024 - predict_length):].unsqueeze(0)
            input_length = len(input_ids[0])
        else:
            input_ids = None
            input_length = 0
        outputs = self.model.generate(input_ids=input_ids,
                                      max_length=input_length + predict_length,
                                      top_p=top_p,
                                      top_k=top_k,
                                      temperature=temperature,
                                      do_sample=do_sample,
                                      num_return_sequences=num)
        return [
            self.tokenizer.decode(
                outputs[i][input_length:], clean_up_tokenization_spaces=True, skip_special_tokens=True)
            for i in range(num)
        ]
