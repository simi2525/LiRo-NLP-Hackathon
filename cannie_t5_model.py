from typing import Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers import AdamW
from transformers import CaninePreTrainedModel, \
    CanineModel, MT5ForConditionalGeneration
from transformers.modeling_outputs import TokenClassifierOutput

from transformers import AutoModel

LR = 1e-4

PRETRAINED_MODELS_CACHE = None

class CanineForTokenClassificationCustom(CaninePreTrainedModel):
    def __init__(self, config, cached_path=None):
        super().__init__(config)
        self.cached_path = cached_path
        self.num_labels = config.num_labels

        self.canine = CanineModel(config)
        # self.t5 = MT5ForConditionalGeneration.from_pretrained('iliemihai/mt5-base-romanian-diacritics').encoder
        self.t5 = AutoModel.from_pretrained("readerbench/RoBERT-small")

        # for param in self.canine.parameters():
        #     param.requires_grad = False
        # for param in self.t5.parameters():
        #     param.requires_grad = False

        self.bert_dropout = nn.Dropout(p=0.2)
        self.canine_dropout = nn.Dropout(p=0.2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size + self.t5.encoder.config.hidden_size, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.classifier_final = nn.Linear(config.hidden_size + self.t5.encoder.config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

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
            t5_input_ids=None,
            t5_attention_mask=None,
            t5_char_tokens=None,

    ) -> Union[Tuple, TokenClassifierOutput]:

        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if PRETRAINED_MODELS_CACHE is not None and id is not None and id[0] in PRETRAINED_MODELS_CACHE:
            outputs = torch.stack([PRETRAINED_MODELS_CACHE[i]["canine"] for i in id]).cuda()
            t5_out = torch.stack([PRETRAINED_MODELS_CACHE[i]["t5"] for i in id]).cuda()
        else:
            outputs = self.canine(
                input_ids=canine_input_ids,
                attention_mask=canine_attention_mask,
                token_type_ids=canine_token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            t5_out = self.t5(input_ids=t5_input_ids, attention_mask=t5_attention_mask)["last_hidden_state"]

            if PRETRAINED_MODELS_CACHE is not None:
                for i, c, t in zip(id, outputs[0], t5_out):
                    PRETRAINED_MODELS_CACHE[i] = {"canine": c.cpu(), "t5": t.cpu()}

        canine_out = outputs[0]
        BS, seq_length_t5, T5_embed = t5_out.shape
        canine_embed = canine_out.shape[-1]
        # print("SHAPPEE" , sequence_output.shape)
        # print("CHAR TOKENS", t5_char_tokens.shape)

        canine_out = canine_out.view(-1, canine_embed)
        t5_out = t5_out.view(-1, T5_embed)
        t5_char_tokens = t5_char_tokens + (torch.arange(BS, device="cuda") * seq_length_t5).unsqueeze(-1)
        char_t5_tokens = t5_out[t5_char_tokens.flatten()]

        sequence_output = torch.concat((canine_out, char_t5_tokens), dim=-1)
        sequence_output = sequence_output.view(BS, -1, canine_embed + T5_embed)
        sequence_output = self.transformer_encoder(sequence_output)

        # sequence_output = self.dropout(sequence_output)
        logits = self.classifier_final(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none').cuda()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss * canine_attention_mask.flatten()
            loss = loss.sum() / (canine_attention_mask.sum() + 1e-15)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CanineReviewClassifier(pl.LightningModule):
    def __init__(self, labels, id2label, label2id):
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
