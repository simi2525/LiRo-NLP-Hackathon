import pytorch_lightning as pl
from transformers import CanineForSequenceClassification, AdamW
import torch.nn as nn
from cannie_model import CanineForTokenClassificationCustom

class CanineReviewClassifier(pl.LightningModule):
    def __init__(self):
        super(CanineReviewClassifier, self).__init__()
        self.model = CanineForTokenClassificationCustom.from_pretrained('google/canine-s', 
                                                                     num_labels=len(labels),
                                                                     id2label=id2label,
                                                                     label2id=label2id)
    # def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
    def forward(self, id, canine_input_ids, canine_token_type_ids, canine_attention_mask, t5_char_tokens,t5_input_ids, t5_attention_mask, labels=None):
        # 'id', 'canine_input_ids', 'canine_token_type_ids', 'canine_attention_mask', "t5_char_tokens",'t5_input_ids','t5_attention_mask', 'labels'
        outputs = self.model(
            id=id, 
            canine_input_ids=canine_input_ids, 
            canine_token_type_ids=canine_token_type_ids, 
            canine_attention_mask=canine_attention_mask, 
            t5_char_tokens=t5_char_tokens,
            t5_input_ids=t5_input_ids,
            t5_attention_mask=t5_attention_mask,
            labels=labels
        )

        return outputs
        
    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits

        predictions = logits.argmax(-1)
        correct = (predictions == batch['labels']).sum().item()
        accuracy = correct/batch['canine_input_ids'].shape[0]

        return loss, accuracy
      
    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     

        return loss

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        return AdamW(self.parameters(), lr=LR)

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return test_dataloader