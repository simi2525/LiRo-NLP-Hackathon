import numpy as np
from transformers import AutoTokenizer
import re

class PreprocessingUtils():
    def __init__(self,percentage_diacritics_removed,max_length):
        self.percentage_diacritics_removed = percentage_diacritics_removed
        self.max_length = max_length
        
        self.labels = [
            "no_diac", 
            "virgula",
            "caciulita",
            "cupa"
        ]

        self.chars_with_virgula = ['ș','ț','Ș','Ț']
        self.chars_can_have_virgula = ['s','t','S','T']
        self.chars_with_caciulita = ['â','î','Â','Î']
        self.chars_can_have_caciulita = ['a','i','A','I']
        self.chars_with_cupa = ['ă','Ă']
        self.chars_can_have_cupa = ['a','A']
        self.chars_can_have_diacritics = ['a','t','i','s', 'A', 'T', 'I', 'S']
        self.chars_with_diacritics = ['ă','â','î','ș', 'ț', 'Ă', 'Â', 'Î', 'Ș', 'Ț']

        self.id2label = {idx:label for idx, label in enumerate(self.labels)}
        self.label2id = {label:idx for idx, label in enumerate(self.labels)}
        
        self.canine_tokenizer = AutoTokenizer.from_pretrained("google/canine-c")
        self.bert_tokenizer = AutoTokenizer.from_pretrained("readerbench/RoBERT-base")

    def char_label2char(self, c, l):
        if c in self.chars_with_diacritics:
            return c
        if c not in self.chars_can_have_diacritics:
            return c
        
        
        if l == self.label2id['virgula'] and c in self.chars_can_have_virgula:
            return self.chars_with_virgula[self.chars_can_have_virgula.index(c)]
    
        if l == self.label2id['caciulita'] and c in self.chars_can_have_caciulita:
            return self.chars_with_caciulita[self.chars_can_have_caciulita.index(c)]
        
        if l == self.label2id['cupa'] and c in self.chars_can_have_cupa:
            return self.chars_with_cupa[self.chars_can_have_cupa.index(c)]
        
        return c
        

    def preprocess_all(self, batch):
        batch = self.preprocess_batch_group_1(batch)
        batch = self.canine_tokenize_input(batch)
        batch = self.pad_attention_mask(batch)
        
        batch['canine_input_ids'] = batch.pop('input_ids')
        batch['canine_token_type_ids'] = batch.pop('token_type_ids')
        batch['canine_attention_mask'] = batch.pop('attention_mask')
        
        batch = self.bert_char2tokens(batch)
        batch = self.bert_tokenize_input(batch)
        batch = self.pad_bert_tokens(batch)
        
        batch['bert_input_ids'] = batch.pop('input_ids')
        batch['bert_attention_mask'] = batch.pop('attention_mask')
        
        batch = self.preprocess_batch_group_2(batch)
        
        return batch
    
    
    def preprocess_batch_group_1(self, examples):
        def remove_diacritics(input_txt):
            diac_map = {'ț': 't', 'ș': 's', 'Ț': 'T', 'Ș': 'S', 'Ă': 'A', 'ă': 'a', 'Â': 'A', 'â': 'a', 'Î': 'I', 'î': 'i'}
            diacritic_positions = [m.start() for m in re.finditer('ț|ș|Ț|Ș|Ă|ă|Â|â|Î|î', input_txt)]
            to_remove_diacritic_positions = np.random.choice(diacritic_positions, int(len(diacritic_positions) * self.percentage_diacritics_removed), replace=False)
            for i in range(len(to_remove_diacritic_positions)):
                input_txt = input_txt[:to_remove_diacritic_positions[i]]+ diac_map[input_txt[to_remove_diacritic_positions[i]]] + input_txt[to_remove_diacritic_positions[i]+1:]
            return input_txt
        def make_a_l(lbl):
            result = []
            for s in lbl:
                if s in self.chars_with_virgula:
                    result.append(self.label2id['virgula'])
                elif s in self.chars_with_caciulita:
                    result.append(self.label2id['caciulita'])
                elif s in self.chars_with_cupa:
                    result.append(self.label2id['cupa'])
                else:
                    result.append(self.label2id["no_diac"])
            return result
        
        def make_im(input_txt):
            return list([0 if e not in self.chars_can_have_diacritics or e in self.chars_with_diacritics else 1 for e in input_txt])
        
        input_aux = []
        label_aux = []
        input_mask_aux = []
        
        for i in range(len(examples['labels'])):
            input_aux.append(remove_diacritics(examples['labels'][i]))
            label_aux.append(make_a_l(examples['labels'][i]))
            input_mask_aux.append(make_im(input_aux[i]))
            
        examples['input'] = input_aux
        examples['labels'] = label_aux
        examples["input_mask"] = input_mask_aux
        return examples
        
    def preprocess_batch_group_2(self, examples):
        def sample_pad_labels(example):
            return [0] + example + [0] + [0] * (self.max_length - len(example) - 2)
        def sample_truncate_canine_attention_mask(example):
            return example[:self.max_length]
        def sample_truncate_bert_char_token(example):
            return example[:self.max_length]
        
        labels_aux = []
        canine_attention_mask_aux = []
        bert_char_tokens_aux = []
        
        for i in range(len(examples['labels'])):
            labels_aux.append(sample_pad_labels(examples['labels'][i])[:self.max_length])
            canine_attention_mask_aux.append(sample_truncate_canine_attention_mask(examples['canine_attention_mask'][i]))
            bert_char_tokens_aux.append(sample_truncate_bert_char_token(examples['bert_char_tokens'][i]))
        
        examples['labels'] = labels_aux
        examples['canine_attention_mask'] = canine_attention_mask_aux
        examples['bert_char_tokens'] = bert_char_tokens_aux
        
        return examples
    
    
    def canine_tokenize_input(self,examples):
        return examples | self.canine_tokenizer(examples['input'], padding="max_length", truncation=True, max_length=self.max_length)
    
    def bert_tokenize_input(self,examples):
        return examples | self.bert_tokenizer(examples['input'], padding="max_length", truncation=True, max_length=self.max_length)
    
    def bert_char2tokens(self,examples):
        def sample_char2tokens(example):
          return self.char2token(example)
        examples["bert_char_tokens"] = [sample_char2tokens(l) for l in examples["input"]]
        return examples
    
    def pad_attention_mask(self, examples):
        def sample_pad_attention_mask(example):
            return [0] + example + [0] + [0] * (self.max_length - len(example) - 2)
        examples["attention_mask"] = [sample_pad_attention_mask(l) for l in examples["input_mask"]]
        return examples
    
    def pad_bert_tokens(self, examples):
        def sample_pad_bert_tokens(example):
            return [0] + example + [0] + [0] * (self.max_length - len(example) - 2)
        examples["bert_char_tokens"] = [sample_pad_bert_tokens(l) for l in examples["bert_char_tokens"]]
        return examples
    
    def char2token(self,input_text):
        input_text = input_text.lower()
        bert_tokenized_input = self.bert_tokenizer.tokenize(input_text)

        bert_tokenized_input = [b.replace("#", '') for b in bert_tokenized_input]
        token_idx = [[i] * len(tok) for i,tok in enumerate(bert_tokenized_input)]
        token_idx = [item for sublist in token_idx for item in sublist]
        bert_tokenized_input = [item for sublist in bert_tokenized_input for item in sublist]

        result = [-1] * len(input_text)
        curr_len = 0
        for token_char_idx, token_char  in zip(token_idx, bert_tokenized_input):
            char_idx = input_text.find(token_char)
            input_text = input_text[char_idx+1:]
            result[curr_len + char_idx] = token_char_idx
            curr_len += (char_idx + 1)

        for i in range(len(result)):
            if result[i] == -1:
                j = min(i + 1, len(result) - 1)
                while j < len(result) - 1 and result[j] == -1:
                    j += 1
                if result[j] == -1:
                    j = i-1
                result[i] = result[j]
                
        return result