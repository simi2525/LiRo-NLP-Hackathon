import numpy as np
import torch
import re
from canine_bert_model import DiacCanineBertTokenClassification
from utils import PreprocessingUtils
from eval_total import Evaluator
from pystardict import Dictionary
import os

from datasets import load_dataset, Dataset

dicts_dir = "dictionaries/"
dicts = []
dicts.append(Dictionary(os.path.join(dicts_dir, '01.dex09-2009','dex09-2009')))
dicts.append(Dictionary(os.path.join(dicts_dir, '02.mdn-2000+2008/mdn00-2000','mdn00-2000')))
dicts.append(Dictionary(os.path.join(dicts_dir, '02.mdn-2000+2008/mdn08-2008','mdn08-2008')))
dicts.append(Dictionary(os.path.join(dicts_dir, '03.dlrlc-1955-1957','dlrlc-1955-1957')))
dicts.append(Dictionary(os.path.join(dicts_dir, '04.dulr6-1929','dulr6-1929')))

if __name__ == '__main__':

    max_length = 256
    percentage_diacritics_removed = 1.0
    device = 'cuda'

    utils = PreprocessingUtils(percentage_diacritics_removed=percentage_diacritics_removed, max_length=max_length)


    hidden_text = file1 = open('diac_hidden_1k.txt', 'r',encoding='utf-8').readlines()
    # hidden_text = file1 = open('test_long_text.txt', 'r',encoding='utf-8').readlines()
    hidden_ds = {"labels": []}

    punctuations = '[\?!,\.:;\s]'
    splitting_points = []
    splitting_points_space = []
    aux_hidden_text = []
    for line in hidden_text:
        i = 0
        while i < len(line):
            aux_line = line[i:i+max_length-2]
            if len(line)-i > max_length-2 and not ' ' in line[i+max_length-3:i+max_length-1]:
                m = re.search(punctuations, aux_line)
                aux_index_position = [i.end() for i in re.finditer(punctuations,aux_line)]
                if len(aux_index_position) > 0:
                    last_punctuation = aux_index_position[-1]
                else:
                    last_punctuation = max_length - 2
                    
                aux_line = aux_line[:last_punctuation]
                i += last_punctuation
            else:
                i += max_length-2
            
            hidden_ds["labels"].append(aux_line.strip())
            aux_hidden_text.append(aux_line.strip())
            
            if i<len(line):
                splitting_points.append(len(hidden_ds["labels"])-1)
                if line[i-1] == ' ':
                    splitting_points_space.append(len(hidden_ds["labels"])-1)
                
    hidden_text = aux_hidden_text

    hidden_ds = Dataset.from_dict(hidden_ds)
    hidden_ds = hidden_ds.map(utils.preprocess_all, batched=True, load_from_cache_file=False)
    hidden_ds.set_format(type="torch", columns=['canine_input_ids', 'canine_token_type_ids', 'canine_attention_mask', "bert_char_tokens",'bert_input_ids','bert_attention_mask', 'labels'])

    # model = DiacCanineBertTokenClassification(num_labels=len(utils.labels)).to(device)
    model = DiacCanineBertTokenClassification.load_from_checkpoint('checkpoints2/epoch=2-step=117189.ckpt', num_labels=len(utils.labels), strict=True).to(device)
    model.eval()



    predictions = []
    for i in range(len(hidden_text)):
        outputs = model(
            canine_input_ids=hidden_ds[i]['canine_input_ids'].unsqueeze(0).to(device),
            canine_token_type_ids=hidden_ds[i]['canine_token_type_ids'].unsqueeze(0).to(device),
            canine_attention_mask=hidden_ds[i]['canine_attention_mask'].unsqueeze(0).to(device),
            bert_char_tokens=hidden_ds[i]['bert_char_tokens'].unsqueeze(0).to(device),
            bert_input_ids=hidden_ds[i]['bert_input_ids'].unsqueeze(0).to(device),
            bert_attention_mask=hidden_ds[i]['bert_attention_mask'].unsqueeze(0).to(device),
            labels=hidden_ds[i]['labels'].unsqueeze(0).to(device),
        )
        
        line = Evaluator.remove_diacritics(hidden_text[i])
        classes = outputs.logits.argmax(-1)[0,1:-1]
        
        predictions.append((line, classes))
        

    def class_to_text(i,prediction):
        line, classes = prediction
        
        result = ''
        for j, c in enumerate(line.strip()):
            result+=utils.char_label2char(c, classes[j])
        
        old_string = line.strip().split(' ')
        new_string = result.split(' ')
        for word_position,(old,new) in enumerate(zip(old_string, new_string)):
            if old != new:
                appears_old = []
                for d in dicts:
                    if old.lower() in d:
                        appears_old.append(d[old.lower()])
                appears_new = []
                for d in dicts:
                    if new.lower() in d:
                        appears_new.append(d[new.lower()])
                if len(appears_old) > 0 and len(appears_new) == 0:
                    new_string[word_position] = old_string[word_position]
        line = ' '.join(old_string)
        result = ' '.join(new_string)        
        
        if i in splitting_points:
            if i in splitting_points_space:
                return result + ' ' 
            else:
                return result
        else:
            return result + '\n'
        
    
    result_file = open("diac_hidden_1k_results.txt", "a", encoding='utf-8')
    for i, p in enumerate(predictions):
        result_file.write(class_to_text(i,p))
    result_file.close()
