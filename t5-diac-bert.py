from transformers import MT5ForConditionalGeneration, T5Tokenizer

model = MT5ForConditionalGeneration.from_pretrained('iliemihai/mt5-base-romanian-diacritics')
tokenizer = T5Tokenizer.from_pretrained('iliemihai/mt5-base-romanian-diacritics')

input_text = "A inceput sa ii taie  un fir de par, iar fata sta in fata, tine camasa de in in mana si canta nota SI."
inputs = tokenizer(input_text, max_length=256, truncation=True, return_tensors="pt")

print(tokenizer.tokenize(input_text))
print(tokenizer(input_text)['input_ids'])

def char2token(input_text):
    bert_tokens = tokenizer(input_text)['input_ids']
    bert_tokenized_input = tokenizer.tokenize(input_text)
    char2token_list = []
    concat_tokens = ''.join(bert_tokenized_input)
    
    token_idx = 0
    internal_token_idx = 0
    for i,c in enumerate(input_text):
        if bert_tokenized_input[token_idx][internal_token_idx] == '▁':
            internal_token_idx += 1

        if c == ' ':
            char2token_list.append(bert_tokens[min(token_idx+1,len(bert_tokens)-1)])
        else:
            char2token_list.append(bert_tokens[token_idx])
        if internal_token_idx >= len(bert_tokenized_input[token_idx]):
            token_idx += 1
        # internal_token_idx += 1
    print(char2token_list)
    print(len(input_text))
    print(len(char2token_list))

char2token(input_text)
exit()

print(model.encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]))
exit()

outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
output = tokenizer.decode(outputs[0], skip_special_tokens=True)


print(output)  # this will print "A început să îi taie un fir de păr, iar fata stă în față, ține cămașa de in în mână și cântă nota SI" 