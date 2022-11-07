import numpy as np
from transformers import CanineTokenizer
from transformers import MT5ForConditionalGeneration, T5Tokenizer

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
        self.chars_with_caciulita = ['â','î','Â','Î']
        self.chars_with_cupa = ['ă','Ă']
        self.chars_can_have_diacritics = ['a','t','i','s', 'A', 'T', 'I', 'S']
        self.chars_with_diacritics = ['ă','â','î','ș', 'ț', 'Ă', 'Â', 'Î', 'Ș', 'Ț']

        self.id2label = {idx:label for idx, label in enumerate(self.labels)}
        self.label2id = {label:idx for idx, label in enumerate(self.labels)}
        
        self.cannie_tokenizer = CanineTokenizer.from_pretrained("google/canine-s")
        self.t5_tokenizer = T5Tokenizer.from_pretrained('iliemihai/mt5-base-romanian-diacritics')


    def preprocess_cannie(self, batch):
        batch = self.add_partial_no_diac_input(batch)
        batch = self.make_actual_labels(batch)
        batch = self.make_input_mask(batch)
        batch = self.make_actual_labels(batch)
        batch = self.make_actual_labels(batch)
        batch = self.make_actual_labels(batch)
        batch = self.make_actual_labels(batch)
        batch = self.make_actual_labels(batch)
        batch = self.tokenize_input(batch)
        batch = [self.pad_attention_mask(b) for b in batch]
        return batch
    
    def preprocess_t5(self, batch):
        batch = self.t5_char2tokens(batch)
        batch = [self.t5_char2tokens(b) for b in batch]
        batch = [self.pad_t5_tokens(b) for b in batch]
        batch = self.t5_tokenize_input(batch)
        return batch
        
    def preprocess_cannie_t5(self, batch):
        batch = [self.pad_labels(b) for b in batch]
        batch = [self.truncate_labels(b) for b in batch]
        batch = [self.truncate_t5_char_tokens(b) for b in batch]
        batch = [self.truncate_canine_attention_mask(b) for b in batch]
        return batch

    def add_partial_no_diac_input(self, examples):
        def remove_diacritics(input_txt):
            diac_map = {'ț': 't', 'ș': 's', 'Ț': 'T', 'Ș': 'S', 'Ă': 'A', 'ă': 'a', 'Â': 'A', 'â': 'a', 'Î': 'I', 'î': 'i'}
            diacritic_positions = [m.start() for m in re.finditer('ț|ș|Ț|Ș|Ă|ă|Â|â|Î|î', input_txt)]
            to_remove_diacritic_positions = np.random.choice(diacritic_positions, int(len(diacritic_positions) * self.percentage_diacritics_removed), replace=False)
            for i in range(len(to_remove_diacritic_positions)):
                input_txt = input_txt[:to_remove_diacritic_positions[i]]+ diac_map[input_txt[to_remove_diacritic_positions[i]]] + input_txt[to_remove_diacritic_positions[i]+1:]
            return input_txt
        examples['input'] = [remove_diacritics(input_txt=l) for l in examples["labels"]]
        return examples
    
    
    def make_actual_labels(self,examples):
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
        examples['labels'] = [make_a_l(l) for l in examples["labels"]]
        return examples
    
    def make_input_mask(self,examples):
        def make_im(input_txt):
            return list([0 if e not in self.chars_can_have_diacritics or e in self.chars_with_diacritics else 1 for e in input_txt])
        examples["input_mask"] = [make_im(l) for l in examples["input"]]
        return examples
    
    def cannie_tokenize_input(self,examples):
        return self.cannie_tokenizer(examples['input'], padding="max_length", truncation=True, max_length=self.max_length)
    
    def t5_tokenize_input(self,examples):
        return self.t5_tokenizer(examples['input'], padding="max_length", truncation=True, max_length=self.max_length)
    
    def t5_char2tokens(self,example):
        return {"t5_char_tokens" : self.char2token(example["input"])}
    
    def pad_attention_mask(self, example):
        return {"attention_mask" : 
            [0] 
            + example["input_mask"] 
            + [0]
            + [0] * (len(example["input_ids"]) - len(example["input_mask"]) - 2)
        }
    
    def pad_t5_tokens(self, example):
        return {"t5_char_tokens" : 
            [0] 
            + example["t5_char_tokens"] 
            + [0] 
            + [0] * (len(example["input_ids"]) - len(example["t5_char_tokens"]) - 2)
        }
    
    def pad_labels(self, example):
        return { "labels" :
            [0] 
            + example["labels"] 
            + [0] 
            + [0] * (max_length - len(example["labels"]) - 2)
        }
    
    def truncate_labels(self,example):
        return {"labels": example["labels"][:self.max_length]}
    
    def truncate_t5_char_tokens(self,example):
        return {"t5_char_tokens": example["t5_char_tokens"][:self.max_length]}
    
    def truncate_canine_attention_mask(self,example):
        return {"truncate_canine_attention_mask": example["truncate_canine_attention_mask"][:self.max_length]}
    
    def char2token(self,input_text):
        bert_tokenized_input = self.t5_tokenizer.tokenize(input_text)
        bert_tokenized_input = [b.replace("▁", '') for b in bert_tokenized_input]

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