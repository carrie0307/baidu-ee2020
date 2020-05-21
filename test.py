from transformers import *
import torch
tokenizer = BertTokenizer.from_pretrained('../roberta/vocab.txt',do_lower_case=False)
model = BertModel.from_pretrained(pretrained_model_name_or_path='/home/cuishiyao/roberta')

input_ids = torch.tensor(tokenizer.encode("今天天气很好", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
print(input_ids)
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
print(tokens, len(tokens))
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
print(last_hidden_states.shape)