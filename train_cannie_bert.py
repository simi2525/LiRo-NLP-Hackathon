import numpy as np
import torch
import re
from datasets import load_dataset
from transformers import CanineTokenizer
import re
import matplotlib.pyplot as plt
from tqdm import tqdm


from torch.utils.data import DataLoader

from utils import PreprocessingUtils
import wandb

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

wandb.login()

max_length = 256

percentage_diacritics_removed = 0.95


dataset = load_dataset("dumitrescustefan/diacritic")
dataset["train"] = dataset["train"].select(list(range(1000)))
dataset["validation"] = dataset["validation"].select(list(range(1000)))
train_ds = dataset
train_ds = train_ds.rename_column("text", "labels")

utils = PreprocessingUtils(percentage_diacritics_removed=percentage_diacritics_removed, max_length=max_length)

train_ds = train_ds.map(utils.preprocess_all, batched=True, num_proc=16)


per_label_counts = [4.69565512e+09, 8.99334790e+07, 6.15484390e+07, 1.25212488e+08]

n_samples = sum(per_label_counts)
per_label_weights = [n_samples / (c * n_samples) for c in per_label_counts]
per_label_weights = [w * 1/max(per_label_weights) for w in per_label_weights]
print("Label weights", per_label_weights)

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