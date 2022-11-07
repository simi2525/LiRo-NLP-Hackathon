import numpy as np
import torch
import re
from datasets import load_dataset
from transformers import CanineTokenizer

from torch.utils.data import DataLoader

from utils import PreprocessingUtils
import wandb

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

wandb.login()

max_length = 1024

percentage_diacritics_removed = 1.0


dataset = load_dataset("dumitrescustefan/diacritic")
dataset["train"] = dataset["train"]#.select(list(range(1000)))
dataset["validation"] = dataset["validation"]#.select(list(range(100)))
train_ds = dataset
train_ds = train_ds.rename_column("text", "labels")

utils = PreprocessingUtils(percentage_diacritics_removed=percentage_diacritics_removed, max_length=max_length)

train_ds = train_ds.map(utils.preprocess_cannie, batched=True, num_proc=16)
# train_ds = train_ds.map(utils.add_partial_no_diac_input, batched=True)
# train_ds = train_ds.map(utils.make_actual_labels, batched=True)
# train_ds = train_ds.map(utils.make_input_mask, batched=True)
# train_ds = train_ds.map(utils.tokenize_input,batched=True)
# train_ds = train_ds.map(utils.pad_attention_mask, num_proc=8)
train_ds = train_ds.rename_columns({"input_ids" : "canine_input_ids", "token_type_ids" : "canine_token_type_ids", "attention_mask" : "canine_attention_mask" })
        

train_ds = train_ds.map(utils.preprocess_t5, batched=True, num_proc=16)
# train_ds = train_ds.map(utils.t5_char2tokens)
# train_ds = train_ds.map(utils.pad_t5_tokens, num_proc=8)
# train_ds = train_ds.map(utils.t5_tokenize_input, batched=True)
train_ds = train_ds.rename_columns({"input_ids" : "t5_input_ids", "attention_mask" : "t5_attention_mask"})

train_ds = train_ds.map(utils.preprocess_cannie_t5, batched=True, num_proc=16)
# train_ds = train_ds.map(utils.pad_labels)
# train_ds = train_ds.map(utils.truncate_labels)
# train_ds = train_ds.map(utils.truncate_t5_char_tokens)
# train_ds = train_ds.map(utils.truncate_canine_attention_mask)

# per_label_counts = [3297059,  313306,  102377,   49724,  118957,  135583]
# n_samples = sum(per_label_counts)
# per_label_weights = [n_samples / (c * n_samples) for c in per_label_counts]
# print("Label weights", per_label_weights)

test_ds = train_ds["validation"]
train_ds = train_ds["train"]

s = train_ds[0]
for key, value in s.items():
    print(key)
    if isinstance(value, list):
        print(len(value))

exit()

train_ds.set_format(type="torch", columns=['id', 'canine_input_ids', 'canine_token_type_ids', 'canine_attention_mask', "t5_char_tokens",'t5_input_ids','t5_attention_mask', 'labels'])
test_ds.set_format(type="torch", columns=['id', 'canine_input_ids', 'canine_token_type_ids', 'canine_attention_mask', "t5_char_tokens",'t5_input_ids','t5_attention_mask', 'labels'])

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 100

train_dataloader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, drop_last=False)

checkpoint_callback = ModelCheckpoint(dirpath="checkpoints_t5", save_top_k=-1, monitor="validation_loss")
# model = CanineReviewClassifier()
model = CanineReviewClassifier.load_from_checkpoint('cannie_v4_checkpoints/epoch=3-step=3748.ckpt', strict=False)
wandb_logger = WandbLogger(name='total_diacricitc_t5_cannie', project='CANINE')
trainer = Trainer(accelerator='gpu',devices=1, logger=wandb_logger, callbacks=[checkpoint_callback], max_epochs=EPOCHS, )
trainer.fit(model)