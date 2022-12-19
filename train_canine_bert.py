import wandb
from datetime import timedelta
from datasets import load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from canine_bert_model import DiacCanineBertTokenClassification
import numpy as np
from utils import PreprocessingUtils

CACHE_PREPROCESSING = True
if __name__ == '__main__':
    wandb.login()

    max_length = 256

    percentage_diacritics_removed = 1.0

    dataset = load_dataset("dumitrescustefan/diacritic")
    dataset["train"] = dataset["train"].shuffle().select(list(range(10000000)))
    dataset["validation"] = dataset["validation"]#.select(list(range(1000)))
    train_ds = dataset
    train_ds = train_ds.rename_column("text", "labels")

    utils = PreprocessingUtils(percentage_diacritics_removed=percentage_diacritics_removed, max_length=max_length)
    train_ds = train_ds.map(utils.preprocess_all, batched=True, num_proc=16,  load_from_cache_file=CACHE_PREPROCESSING)

    # print(utils.avg_word_len/utils.word_counter)
    # avg_word_len 3.34

    per_label_counts = [4.69565512e+09, 8.99334790e+07, 6.15484390e+07, 1.25212488e+08]
    per_label_counts = [np.log(x) for x in per_label_counts]
    per_label_counts = [x - 15 for x in per_label_counts]

    n_samples = sum(per_label_counts)
    per_label_weights = [n_samples / (c * n_samples) for c in per_label_counts]
    per_label_weights = [w * 1/max(per_label_weights) for w in per_label_weights]
    print("Label weights", per_label_weights)
    exit()
    # per_label_weights = [1.0, 1.0, 1.0, 1.0]

    test_ds = train_ds["validation"]
    train_ds = train_ds["train"]

    s = train_ds[0]
    for key, value in s.items():
        if isinstance(value, list):
            print(key,len(value))

    train_ds.set_format(type="torch", columns=['id', 'canine_input_ids', 'canine_token_type_ids', 'canine_attention_mask', "bert_char_tokens",'bert_input_ids','bert_attention_mask', 'labels'])
    test_ds.set_format(type="torch", columns=['id', 'canine_input_ids', 'canine_token_type_ids', 'canine_attention_mask', "bert_char_tokens",'bert_input_ids','bert_attention_mask', 'labels'])

    TRAIN_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32
    LR = 1e-6
    EPOCHS = 10

    train_dataloader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, persistent_workers=True)
    test_dataloader = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, drop_last=False, num_workers=4, persistent_workers=True)

    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints_canine_bert", save_top_k=-1, monitor="validation_loss")
    checkpoint_callback_hours = ModelCheckpoint(dirpath="checkpoints_canine_bert_4hours",every_n_train_steps=None,every_n_epochs=None, train_time_interval=timedelta(hours=4), save_top_k=6, monitor="step")
    
    wandb_logger = WandbLogger(name='canine-c_bert-base', project='Diacritic')
    
    # model = DiacCanineBertTokenClassification(num_labels=len(utils.labels), per_label_weights=per_label_weights, lr=LR)
    model = DiacCanineBertTokenClassification.load_from_checkpoint('checkpoints2/epoch=2-step=117189.ckpt', strict=False, num_labels=len(utils.labels), per_label_weights=per_label_weights, lr=LR)
    
    # for param in model.canine.parameters():
    #     param.requires_grad = False
    # for param in model.bert.parameters():
    #     param.requires_grad = False
    
    trainer = Trainer(accelerator='gpu',precision='bf16', amp_backend="native",devices=2, logger=wandb_logger, callbacks=[checkpoint_callback,checkpoint_callback_hours], max_epochs=EPOCHS, accumulate_grad_batches=8)
    
    # trainer.fit(model, train_dataloader, test_dataloader)
    trainer.fit(model, train_dataloader, test_dataloader, ckpt_path='checkpoints2/epoch=2-step=117189.ckpt')