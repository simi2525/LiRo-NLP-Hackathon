from typing import Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import CaninePreTrainedModel, \
    CanineModel, CanineConfig
from transformers.modeling_outputs import TokenClassifierOutput
from torchmetrics import Accuracy
from transformers import AutoModel

PRETRAINED_MODELS_CACHE = None

class DiacCanineBertTokenClassification(pl.LightningModule):
    def __init__(self, num_labels, per_label_weights=None, lr=1e-4):
        super().__init__()
        self.num_labels = num_labels
        if per_label_weights is None:
            per_label_weights = torch.ones(num_labels)
        self.per_label_weights = per_label_weights
        self.lr = lr

        self.canine = AutoModel.from_pretrained('google/canine-c')
        self.bert = AutoModel.from_pretrained("readerbench/RoBERT-base")

        # for param in self.canine.parameters():
        #     param.requires_grad = True
        # for param in self.bert.parameters():
        #     param.requires_grad = True

        self.bert_dropout = nn.Dropout(p=0.2)
        self.canine_dropout = nn.Dropout(p=0.2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.canine.config.hidden_size + self.bert.encoder.config.hidden_size, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2) # TODO Try to get this to 4 or 6 and fit batch size of 64
        self.classifier_final = nn.Linear(self.canine.config.hidden_size + self.bert.encoder.config.hidden_size, self.num_labels)

        # Initialize weights and apply final processing
        self.canine.post_init()
        self.bert.post_init()
        
        
        self.train_metric = Accuracy(num_classes=self.num_labels)
        self.train_metric_per_label = Accuracy(num_classes=self.num_labels, average='none')
        
        self.val_metric = Accuracy(num_classes=self.num_labels)
        self.val_metric_per_label = Accuracy(num_classes=self.num_labels, average='none')

    def forward(
            self,
            canine_input_ids: Optional[torch.LongTensor] = None,
            canine_attention_mask: Optional[torch.FloatTensor] = None,
            canine_token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            id=None,
            bert_input_ids=None,
            bert_attention_mask=None,
            bert_char_tokens=None,

    ) -> Union[Tuple, TokenClassifierOutput]:

        outputs = self.canine(
            input_ids=canine_input_ids,
            attention_mask=canine_attention_mask,
            token_type_ids=canine_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        bert_out = self.bert(input_ids=bert_input_ids, attention_mask=bert_attention_mask)["last_hidden_state"]
        

        canine_out = outputs[0]
        BS, seq_length_bert, Bert_embed = bert_out.shape
        canine_embed = canine_out.shape[-1]

        canine_out = canine_out.view(-1, canine_embed)
        bert_out = bert_out.view(-1, Bert_embed)

        canine_out = self.canine_dropout(canine_out)
        bert_out = self.bert_dropout(bert_out)
        bert_char_tokens = bert_char_tokens + (torch.arange(BS, device=self.device) * seq_length_bert).unsqueeze(-1)
        char_bert_tokens = bert_out[bert_char_tokens.flatten()]

        sequence_output = torch.concat((canine_out, char_bert_tokens), dim=-1)
        sequence_output = sequence_output.view(BS, -1, canine_embed + Bert_embed)
        sequence_output = self.transformer_encoder(sequence_output)

        logits = self.classifier_final(sequence_output)

        loss_fct = CrossEntropyLoss(weight=torch.tensor(self.per_label_weights),reduction='none').to(self.device)
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        loss = loss * canine_attention_mask.flatten()
        loss = loss.sum() / (canine_attention_mask.sum() + 1e-15)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits

        return loss, logits

    def flatten_and_mask(self, outputs):
        not_masked_samples = outputs["mask"].flatten().nonzero().squeeze()
        preds = outputs['preds'].argmax(-1).flatten()[not_masked_samples]
        target = outputs["target"].flatten()[not_masked_samples]
        return preds, target
      
    def training_step(self, batch, batch_idx):
        loss, logits = self.common_step(batch, batch_idx)
        outputs = {'loss': loss, 'preds': logits, 'target': batch["labels"], 'mask':batch["canine_attention_mask"]}
        preds, target = self.flatten_and_mask(outputs)

        self.log("train_loss", outputs["loss"],on_step=True, on_epoch=True)
        self.log('train_acc', self.train_metric(preds, target), on_step=True, on_epoch=True)
        accs = self.train_metric_per_label(preds, target)
        for i, acc in enumerate(accs): # accs : accuracy per class
              self.log(f'train_acc_class_{i}', acc, on_step=True, on_epoch=True)
        
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, logits = self.common_step(batch, batch_idx)
        outputs = {'loss': loss, 'preds': logits, 'target': batch["labels"], 'mask': batch["canine_attention_mask"]}
        preds, target = self.flatten_and_mask(outputs)

        self.log("validation_loss", outputs["loss"], on_step=True, on_epoch=True, sync_dist=True)
        self.log('validation_acc', self.val_metric(preds, target), on_step=True, on_epoch=True, sync_dist=True)
        accs = self.val_metric_per_label(preds, target)
        for i, acc in enumerate(accs): # accs : accuracy per class
              self.log(f'validation_acc_class_{i}', acc, on_step=True, on_epoch=True, sync_dist=True)
        
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        self.train_metric.reset()
        self.train_metric_per_label.reset()

    def validation_epoch_end(self, outputs):
        self.val_metric.reset()
        self.val_metric_per_label.reset()

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        return AdamW(self.parameters(), lr=self.lr)
