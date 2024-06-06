import inspect
import random
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_deepspeed_available
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from torch.nn import CrossEntropyLoss

def to_cuda(batch_, enable_fsdp, local_rank):
    # batch_ to cuda
    for key in batch_.keys():
        if enable_fsdp:
            # print('local_rank', local_rank)
            batch_[key] = batch_[key].to(local_rank)
        else:
            batch_[key] = batch_[key].to('cuda:0')   
    return batch_
  
class MyKRSLTool:
    def __init__(
        self,
        train_config=None,
        local_rank='cuda:0'
    ):
        # beta: bate * KL
        # loss_type: sigmoid, hinge
        # forward_type: separate, concat
        self.is_encoder_decoder = False
        self.label_pad_token_id = -100
        self.forward_type = train_config.forward_type
        self.train_config = train_config
        self.local_rank = local_rank
        self.ignore_index = -100
    
    def concatenated_inputs(self, batch):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}


        for k in batch:
            if k.startswith("chosen"):
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = batch[k]
        for k in batch:
            if k.startswith("rejected"):
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        batch[k],
                    ),
                    dim=0,
                )

        return to_cuda(concatenated_batch, self.train_config.enable_fsdp, self.local_rank)

    def _get_weighted_batch_loss(self, logits, labels, weights, vocab_size):
        """
        Calculate the loss for a batch of logits and labels, adjusted by weights.

        Args:
            logits: Logits from the model.
            labels: Ground truth labels.
            weights: Weights for each token in the batch.
            vocab_size: Size of the vocabulary.

        Returns:
            Weighted loss for the batch.
        """
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute loss with weights
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))

        # reshape to [bs, seq_len - 1]
        loss = loss.view_as(shift_labels)

        # Apply weights
        weighted_loss = loss * weights[..., 1:]

        # ignore mask
        mask = shift_labels != self.ignore_index
        weighted_loss[~mask] = 0

        # calculate average loss per sample
        sample_losses = weighted_loss.sum(dim=1) / mask.sum(dim=1)

        return sample_losses.mean()

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False
    ): 
        chosen_weights = inputs.pop("chosen_weights").to(self.local_rank) if self.train_config.enable_fsdp else inputs.pop("chosen_weights").to('cuda:0')
        rejected_weights = inputs.pop("rejected_weights").to(self.local_rank) if self.train_config.enable_fsdp else inputs.pop("rejected_weights").to('cuda:0')
        # concat rationale and answer in inputs
        # and then respectively calculate rationale loss and answer loss by CE loss(logits and labels)
        concatenated_batch = self.concatenated_inputs(inputs)
        len_chosen = inputs["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            **model_kwargs,
        )
        all_logits = outputs.logits.to(torch.float32)
        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]
        chosen_labels = concatenated_batch["concatenated_labels"][:len_chosen]
        rejected_labels = concatenated_batch["concatenated_labels"][len_chosen:]

        # Compute NLL and adjust rejected_weights
        with torch.no_grad():
            nll = -F.log_softmax(rejected_logits, dim=-1)
            mask = rejected_labels != self.label_pad_token_id
            mask_rejected_labels = torch.where(mask, rejected_labels, torch.tensor(0, device=rejected_labels.device))
            nll = torch.gather(nll, -1, mask_rejected_labels.unsqueeze(-1)).squeeze(-1)
            nll = torch.where(mask, nll, torch.tensor(0.0, device=nll.device))
            nll_mask = nll < self.krsl_nll_threshold
            rejected_weights = torch.where(mask, rejected_weights * nll_mask.float(), torch.tensor(0.0, device=rejected_weights.device))


        chosen_loss = self._get_weighted_batch_loss(chosen_logits, chosen_labels, chosen_weights, vocab_size=model.config.vocab_size)
        rejected_loss = self._get_weighted_batch_loss(rejected_logits, rejected_labels, rejected_weights, vocab_size=model.config.vocab_size)

        loss = chosen_loss + rejected_loss
        metrics = {
            'train_loss': loss.cpu().item(),
            'chosen_loss(desired_words)': chosen_loss.cpu().item(),
            'rejected_loss(undesired_words)': rejected_loss.cpu().item()
        }
        return loss, metrics

class LLMSTTool:
    def __init__(
        self,
        train_config,
        local_rank
    ):
        self.forward_type = train_config.forward_type
        self.train_config = train_config
        self.local_rank = local_rank

    def compute_loss(
        self,
        model,
        inputs,
    ):
        rationale_loss = torch.tensor(-1, dtype=torch.float32)
        answer_loss = torch.tensor(-1, dtype=torch.float32)
        loss = model(**to_cuda(inputs, self.train_config.enable_fsdp, self.local_rank)).loss
        return loss, rationale_loss, answer_loss
    
class LLMWeightSTTool:
    def __init__(
        self,
        train_config,
        local_rank
    ):
        self.forward_type = train_config.forward_type
        self.train_config = train_config
        self.local_rank = local_rank
        self.ignore_index = -100

    def _get_weighted_batch_loss(self, logits, labels, weights, vocab_size):
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute loss with weights
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1)).view_as(shift_labels)

        adjusted_weights = weights.clone()
        adjusted_weights[..., 1:][shift_labels == self.ignore_index] = 0

        # Apply weights
        weighted_loss = loss * adjusted_weights[..., 1:]

        # Group by weights and calculate the average loss for each weight across the batch
        unique_weights = torch.unique(adjusted_weights)
        weight_losses = {weight.item(): [] for weight in unique_weights if weight.item() != 0}

        for weight in unique_weights:
            if weight.item() == 0:  # Skip the padding or ignored tokens
                continue
            # Create a mask for the current weight across the entire batch
            mask = adjusted_weights == weight
            # Calculate the mean loss for the current weight
            mean_loss = torch.sum(weighted_loss * mask) / torch.sum(mask)
            weight_losses[weight.item()].append(mean_loss)

        # Sum the average losses for all weights
        total_loss = torch.sum(torch.stack(list(weight_losses.values())))

        return total_loss

    def compute_loss(
        self,
        model,
        inputs,
    ):
        inputs = to_cuda(inputs, self.train_config.enable_fsdp, self.local_rank)
        weights = inputs.pop('weights')
        rationale_loss = torch.tensor(-1, dtype=torch.float32)
        answer_loss = torch.tensor(-1, dtype=torch.float32)
        outputs = model(**inputs)
        logits = outputs.logits.to(torch.float32)
        loss = self._get_weighted_batch_loss(logits, inputs['labels'], weights, model.config.vocab_size)

        return loss, rationale_loss, answer_loss
  

class LLMSCOTTTool:
    def __init__(
        self,
        train_config,
        local_rank
    ):
        self.forward_type = train_config.forward_type
        self.train_config = train_config
        self.local_rank = local_rank
        self.alpha = train_config.alpha

    def compute_loss(
        self,
        model,
        inputs,
    ):
        pos_batch = {
            "input_ids": inputs['pos_input_ids'],
            "labels": inputs['pos_labels'],
            "attention_mask": inputs['pos_attention_mask'],
        }
        neg_answer_batch = {
            "input_ids": inputs['neg_answer_input_ids'],
            "labels": inputs['neg_answer_labels'],
            "attention_mask": inputs['neg_answer_attention_mask'],
        }
        rationale_loss = torch.tensor(-1, dtype=torch.float32)
        answer_loss = torch.tensor(-1, dtype=torch.float32)
        pos_loss = model(**to_cuda(pos_batch, self.train_config.enable_fsdp, self.local_rank)).loss
        neg_answer_loss = model(**to_cuda(neg_answer_batch, self.train_config.enable_fsdp, self.local_rank)).loss
        loss = self.alpha * neg_answer_loss + (1 - self.alpha) * pos_loss
        return loss, pos_loss, neg_answer_loss

class LLMMTTool:
    def __init__(
        self,
        train_config,
        local_rank,
    ):
        # alpha: loss = alpha * answer_loss + (1 - alpha) * rationale_loss
        # forward_type: separate, concat
        self.is_encoder_decoder = False
        self.label_pad_token_id = -100
        self.alpha = train_config.alpha
        self.forward_type = train_config.forward_type
        self.train_config = train_config
        self.local_rank = local_rank
    
    def concatenated_inputs(self, batch):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'rationale_input_ids' and 'answer_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}


        for k in batch:
            if k.startswith("rationale"):
                concatenated_key = k.replace("rationale", "concatenated")
                concatenated_batch[concatenated_key] = batch[k]
        for k in batch:
            if k.startswith("answer"):
                concatenated_key = k.replace("answer", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        batch[k],
                    ),
                    dim=0,
                )

        return to_cuda(concatenated_batch, self.train_config.enable_fsdp, self.local_rank)

    def _get_batch_loss(self, logits, labels, vocab_size):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return loss

    def compute_loss(
        self,
        model,
        inputs,
    ): 
        if self.forward_type == 'separate':
            batch_rationale = {
                'input_ids': inputs['rationale_input_ids'],
                'labels': inputs['rationale_labels'],
                'attention_mask': inputs['rationale_attention_mask']
            }
            batch_answer = {
                'input_ids': inputs['answer_input_ids'],
                'labels': inputs['answer_labels'],
                'attention_mask': inputs['answer_attention_mask']
            }
            batch_rationale = to_cuda(batch_rationale, self.train_config.enable_fsdp, self.local_rank)
            rationale_loss = model(**batch_rationale).loss
            batch_answer = to_cuda(batch_answer, self.train_config.enable_fsdp, self.local_rank)
            answer_loss = model(**batch_answer).loss
        elif self.forward_type == 'concat':
            # concat rationale and answer in inputs
            # and then respectively calculate rationale loss and answer loss by CE loss(logits and labels)
            concatenated_batch = self.concatenated_inputs(inputs)
            len_rationale = inputs["rationale_labels"].shape[0]

            model_kwargs = (
                {
                    "labels": concatenated_batch["concatenated_labels"],
                    "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
                }
                if self.is_encoder_decoder
                else {}
            )
            outputs = model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                **model_kwargs,
            )
            all_logits = outputs.logits.to(torch.float32)
            rationale_logits = all_logits[:len_rationale]
            answer_logits = all_logits[len_rationale:]
            rationale_labels = concatenated_batch["concatenated_labels"][:len_rationale]
            answer_labels = concatenated_batch["concatenated_labels"][len_rationale:]

            rationale_loss = self._get_batch_loss(rationale_logits, rationale_labels, vocab_size=model.config.vocab_size)
            answer_loss = self._get_batch_loss(answer_logits, answer_labels, vocab_size=model.config.vocab_size)

        loss = (1 - self.alpha) * rationale_loss + self.alpha * answer_loss
        return loss, rationale_loss, answer_loss
    
class LLMCMTTool:
    def __init__(
        self,
        train_config,
        local_rank,
    ):
        # alpha: loss = alpha * answer_loss + (1 - alpha) * rationale_loss
        # forward_type: separate, concat
        self.is_encoder_decoder = False
        self.label_pad_token_id = -100
        self.alpha = train_config.alpha
        self.forward_type = train_config.forward_type
        self.train_config = train_config
        self.local_rank = local_rank
    
    def concatenated_inputs(self, batch):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'rationale_input_ids' and 'llmrationale_answer_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}


        for k in batch:
            if k.startswith("rationale"):
                concatenated_key = k.replace("rationale", "concatenated")
                concatenated_batch[concatenated_key] = batch[k]
        for k in batch:
            if k.startswith("llmrationale_answer"):
                concatenated_key = k.replace("llmrationale_answer", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        batch[k],
                    ),
                    dim=0,
                )

        return to_cuda(concatenated_batch, self.train_config.enable_fsdp, self.local_rank)

    def _get_batch_loss(self, logits, labels, vocab_size):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return loss

    def compute_loss(
        self,
        model,
        inputs,
    ): 
        if self.forward_type == 'separate':
            batch_rationale = {
                'input_ids': inputs['rationale_input_ids'],
                'labels': inputs['rationale_labels'],
                'attention_mask': inputs['rationale_attention_mask']
            }
            batch_rationale_answer = {
                'input_ids': inputs['llmrationale_answer_input_ids'],
                'labels': inputs['llmrationale_answer_labels'],
                'attention_mask': inputs['llmrationale_answer_attention_mask']
            }
            batch_rationale = to_cuda(batch_rationale, self.train_config.enable_fsdp, self.local_rank)
            rationale_loss = model(**batch_rationale).loss
            batch_rationale_answer = to_cuda(batch_rationale_answer, self.train_config.enable_fsdp, self.local_rank)
            answer_loss = model(**batch_rationale_answer).loss
        elif self.forward_type == 'concat':
            # concat rationale and answer in inputs
            # and then respectively calculate rationale loss and answer loss by CE loss(logits and labels)
            concatenated_batch = self.concatenated_inputs(inputs)
            len_rationale = inputs["rationale_labels"].shape[0]

            model_kwargs = (
                {
                    "labels": concatenated_batch["concatenated_labels"],
                    "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
                }
                if self.is_encoder_decoder
                else {}
            )
            outputs = model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                **model_kwargs,
            )
            all_logits = outputs.logits.to(torch.float32)
            rationale_logits = all_logits[:len_rationale]
            llmrationale_answer_logits = all_logits[len_rationale:]
            rationale_labels = concatenated_batch["concatenated_labels"][:len_rationale]
            llmrationale_answer_labels = concatenated_batch["concatenated_labels"][len_rationale:]

            rationale_loss = self._get_batch_loss(rationale_logits, rationale_labels, vocab_size=model.config.vocab_size)
            answer_loss = self._get_batch_loss(llmrationale_answer_logits, llmrationale_answer_labels, vocab_size=model.config.vocab_size)

        loss = (1 - self.alpha) * rationale_loss + self.alpha * answer_loss
        return loss, rationale_loss, answer_loss