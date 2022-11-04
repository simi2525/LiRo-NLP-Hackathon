from datasets import load_dataset
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, BertTokenizer, \
    DataCollatorWithPadding, DataCollatorForSeq2Seq, Seq2SeqTrainer, AutoConfig, AutoModelForSeq2SeqLM, \
    Seq2SeqTrainingArguments

tokenizer = BertTokenizer.from_pretrained("readerbench/RoBERT-small")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
# tokenizer.add_tokens(" < sep>")
text_column = "input"
summary_column = "labels"


def _add_special_tokens(example):
    example['text'] = example['text'] + "{sep_token}"
    example['text'] = example['text'].replace("{sep_token}", " < sep>")
    return example

def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[summary_column]
    # model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
    model_inputs = tokenizer(inputs, truncation=True)


    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        # labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
        labels = tokenizer(targets, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    # if padding == "max_length" and data_args.ignore_pad_token_for_loss:
    #     labels["input_ids"] = [
    #         [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    #     ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def tokenize_function(example):
    tokenized_input = tokenizer(example["input"], truncation=True)
    targets = example["text"]
    labels = tokenizer(targets, truncation=True)
    # tokenized_output = tokenizer(example["text"], truncation=True)
    # tokenized_output = {"text_"+key : value for key,value in tokenized_output.items()}
    tokenized_input["labels"] = labels["input_ids"]
    return tokenized_input
    # return {**tokenized_input}


def remove_diacritics(input_txt):
    input_txt = input_txt.replace("ă", "a")
    input_txt = input_txt.replace("î", "i")
    input_txt = input_txt.replace("ș", "s")
    input_txt = input_txt.replace("ț", "s")
    input_txt = input_txt.replace("â", "a")
    return input_txt


def add_no_diac_input(example):
    return {"input": remove_diacritics(input_txt=example["text"])}


if __name__ == "__main__":
    dataset = load_dataset("dumitrescustefan/diacritic")
    # dataset = dataset.select(list(range(100)))
    dataset["train"] = dataset["train"].select(list(range(100)))
    dataset["validation"] = dataset["validation"].select(list(range(100)))
    dataset = dataset.map(add_no_diac_input)

    dataset = dataset.rename_column("text", "labels")
    # dataset = dataset.map(_add_special_tokens)

    print(dataset["train"][0])

    column_names = dataset["train"].column_names

    train_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        # load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on train dataset",
    )

    eval_dataset = dataset["validation"].map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        # load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on validation dataset",
    )
    # config = AutoConfig.from_pretrained("readerbench/RoBERT-small")

    encoder = BertGenerationEncoder.from_pretrained("readerbench/RoBERT-small", bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
    # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
    decoder = BertGenerationDecoder.from_pretrained(
        # "bert-large-uncased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102
        "readerbench/RoBERT-small", add_cross_attention=True, is_decoder=True, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id
    )
    # config =
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)



    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    label_pad_token_id = tokenizer.pad_token_id
    print("Pad token id", label_pad_token_id)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        # pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    training_args = Seq2SeqTrainingArguments(
        output_dir="./",
        learning_rate=5e-5,
        # evaluation_strategy="steps",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        overwrite_output_dir=True,
        save_total_limit=3,
        # fp16=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    train_result = trainer.train()

    # dataset = dataset.map(tokenize_function, batched=True)
    # print(dataset["train"][0])
    #
    # encoder = BertGenerationEncoder.from_pretrained("readerbench/RoBERT-small", bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
    # # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
    # decoder = BertGenerationDecoder.from_pretrained(
    #     # "bert-large-uncased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102
    #     "readerbench/RoBERT-small", add_cross_attention=True, is_decoder=True, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id
    # )
    # bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    # data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    #
    # samples = dataset["train"][:8]
    # samples = {k: v for k, v in samples.items() if k not in ["id", "text", "input"]}
    # batch = data_collator(samples)
    # print(batch)
    # print({k: v.shape for k, v in batch.items()})
    #
    #
