import numpy as np
import torch
import re
from canine_bert_model import DiacCanineBertTokenClassification
from utils import PreprocessingUtils
from eval_total import Evaluator

from datasets import load_dataset, Dataset


max_length = 256
percentage_diacritics_removed = 1.0

utils = PreprocessingUtils(percentage_diacritics_removed=percentage_diacritics_removed, max_length=max_length)


hidden_text = file1 = open('diac_hidden_1k.txt', 'r').readlines()
hidden_ds = {"labels": []}

for line in hidden_text:
    hidden_ds["labels"].append(line.strip())

hidden_ds = Dataset.from_dict(hidden_ds)
hidden_ds = hidden_ds.map(utils.preprocess_all, batched=True, load_from_cache_file=False)
hidden_ds.set_format(type="torch", columns=['canine_input_ids', 'canine_token_type_ids', 'canine_attention_mask', "bert_char_tokens",'bert_input_ids','bert_attention_mask', 'labels'])

# model = DiacCanineBertTokenClassification(num_labels=len(utils.labels)).to('cpu')
model = DiacCanineBertTokenClassification.load_from_checkpoint('checkpoints/epoch=1-step=156250.ckpt', num_labels=len(utils.labels), strict=True).to('cpu')
model.eval()


result_file = open("diac_hidden_1k_results.txt", "a")

for i, line in enumerate(hidden_text):
    outputs = model(
        canine_input_ids=hidden_ds[i]['canine_input_ids'].unsqueeze(0).to('cpu'),
        canine_token_type_ids=hidden_ds[i]['canine_token_type_ids'].unsqueeze(0).to('cpu'),
        canine_attention_mask=hidden_ds[i]['canine_attention_mask'].unsqueeze(0).to('cpu'),
        bert_char_tokens=hidden_ds[i]['bert_char_tokens'].unsqueeze(0).to('cpu'),
        bert_input_ids=hidden_ds[i]['bert_input_ids'].unsqueeze(0).to('cpu'),
        bert_attention_mask=hidden_ds[i]['bert_attention_mask'].unsqueeze(0).to('cpu'),
        labels=hidden_ds[i]['labels'].unsqueeze(0).to('cpu'),
    )
    line = Evaluator.remove_diacritics(line)
    classes = outputs.logits.argmax(-1)[0,1:-1]
    # print(outputs.logits[0,:len(line.strip())])
    # exit()
    result = ''
    for j, c in enumerate(line.strip()):
        result+=utils.char_label2char(c, classes[j].item())
    result_file.write(result+'\n')
    result_file.flush()

result_file.close()


# model = DiacCanineBertTokenClassification.load_from_checkpoint('checkpoints/epoch3.ckpt', strict=True, num_labels=len(utils.labels))
