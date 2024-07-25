import os
import time
import yaml
from pathlib import Path
from pkg_resources import packaging
import random
import copy
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import AutoTokenizer
from contextlib import nullcontext
import sys
sys.path.append('../..')
from src.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint
from src.policies import fpSixteen,bfSixteen_mixed, get_llama_wrapper, get_mistral_wrapper
from src.utils.memory_utils import MemoryTrace
from src.utils.inference_utils import eval_inference
from src.model_checkpointing import *
from src.utils.method_utils import *
from dataclasses import dataclass, asdict
import json

def set_tokenizer_params(tokenizer: AutoTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2**20)

def train(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None, current_epoch=-1, dataset_config=None, inference_config=None, train_cfg_ins=None, inference_cfg_ins=None, ref_model=None):
    """
    Trains the model on the given dataloader
    
    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons
        inference_config: ...
    
    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler() 
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"]) 
    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    train_rationale_loss = []
    train_answer_loss = []
    val_prep = []
    val_loss =[]
    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    best_val_acc = float(-1)
    best_acc_epoch = -1
    train_result_file = None
    if ref_model:
        ref_model.eval()

    krsl = False
    if 'llmweightst' in train_config.dataset:
        llmweightst_tool = LLMWeightSTTool(train_config, local_rank)
    elif 'llmscott' in train_config.dataset:
        llmscott_tool = LLMSCOTTTool(train_config, local_rank)
    elif 'krsl' in train_config.dataset:
        krsl = True
        krsl_tool = MyKRSLTool(train_config, local_rank)
    elif 'llmcmt' in train_config.dataset:
        llmcmt_tool = LLMCMTTool(train_config, local_rank)
    elif 'llmmt' in train_config.dataset:
        llmmt_tool = LLMMTTool(train_config, local_rank)
    elif 'llmst' in train_config.dataset or 'llmstepst' in train_config.dataset:
        llmst_tool = LLMSTTool(train_config, local_rank)
    elif 'bbh_dataset' in train_config.dataset:
        pass
    else:
        raise ValueError('Unknown train_config.dataset')

    for epoch in range(current_epoch, train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_rationale_loss = 0.0
            total_answer_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length)
            for step, batch in enumerate(train_dataloader): 
                if 'llmweightst' in dataset_config.dataset:
                    batch_ = {
                        'input_ids': batch['input_ids'],
                        'labels': batch['labels'],
                        'attention_mask': batch['attention_mask'],
                        'weights': batch['weights']
                    }
                elif 'llmscott' in dataset_config.dataset:
                    batch_ = {
                        'pos_input_ids': batch['pos_input_ids'],
                        'pos_labels': batch['pos_labels'],
                        'pos_attention_mask': batch['pos_attention_mask'],
                        'neg_answer_input_ids': batch['neg_answer_input_ids'],
                        'neg_answer_labels': batch['neg_answer_labels'],
                        'neg_answer_attention_mask': batch['neg_answer_attention_mask'],
                    }
                elif 'krsl' in dataset_config.dataset:
                    chosen_tokens = {
                        'chosen_input_ids': batch['chosen_input_ids'],
                        'chosen_labels': batch['chosen_labels'],
                        'chosen_attention_mask': batch['chosen_attention_mask']
                    }
                    rejected_tokens = {
                        'rejected_input_ids': batch['rejected_input_ids'],
                        'rejected_labels': batch['rejected_labels'],
                        'rejected_attention_mask': batch['rejected_attention_mask']
                    }
                    chosen_weights = batch['chosen_weights']
                    rejected_weights = batch['rejected_weights']
                elif 'llmcmt' in dataset_config.dataset:
                    batch_rationale = {
                        'rationale_input_ids': batch['rationale_input_ids'],
                        'rationale_labels': batch['rationale_labels'],
                        'rationale_attention_mask': batch['rationale_attention_mask']
                    }
                    batch_rationale_answer = {
                        'llmrationale_answer_input_ids': batch['llmrationale_answer_input_ids'],
                        'llmrationale_answer_labels': batch['llmrationale_answer_labels'],
                        'llmrationale_answer_attention_mask': batch['llmrationale_answer_attention_mask']
                    }
                elif 'llmst' in dataset_config.dataset or 'llmstepst' in dataset_config.dataset:
                    batch_ = {
                        'input_ids': batch['input_ids'],
                        'labels': batch['labels'],
                        'attention_mask': batch['attention_mask']
                    }
                elif 'llmmt' in dataset_config.dataset:
                    batch_rationale = {
                        'rationale_input_ids': batch['rationale_input_ids'],
                        'rationale_labels': batch['rationale_labels'],
                        'rationale_attention_mask': batch['rationale_attention_mask']
                    }
                    batch_answer = {
                        'answer_input_ids': batch['answer_input_ids'],
                        'answer_labels': batch['answer_labels'],
                        'answer_attention_mask': batch['answer_attention_mask']
                    }
                else:
                    ## sft
                    batch_ = {
                        'input_ids': batch['input_ids'],
                        'labels': batch['labels'],
                        'attention_mask': batch['attention_mask']
                    }


                with autocast():
                    if 'llmweightst' in dataset_config.dataset:
                        loss, rationale_loss, answer_loss = llmweightst_tool.compute_loss(model, batch_)
                    elif 'llmscott' in dataset_config.dataset:
                        loss, rationale_loss, answer_loss = llmscott_tool.compute_loss(model, batch_)
                    elif 'llmst' in dataset_config.dataset or 'llmstepst' in train_config.dataset:
                        loss, rationale_loss, answer_loss = llmst_tool.compute_loss(model, batch_)
                    elif 'llmmt' in dataset_config.dataset:
                        batch_combined = {**batch_rationale, **batch_answer}
                        loss, rationale_loss, answer_loss = llmmt_tool.compute_loss(model, batch_combined)
                    elif 'krsl' in dataset_config.dataset:
                        batch_combined = {**chosen_tokens, **rejected_tokens,
                                          'chosen_weights': chosen_weights,
                                          'rejected_weights': rejected_weights}
                        loss, metrics = krsl_tool.compute_loss(model, batch_combined, return_outputs=True)

                        rationale_loss = torch.tensor(-1, dtype=torch.float32)
                        answer_loss = torch.tensor(-1, dtype=torch.float32)
                    elif 'llmcmt' in dataset_config.dataset:
                        batch_combined = {**batch_rationale, **batch_rationale_answer}
                        loss, rationale_loss, answer_loss = llmcmt_tool.compute_loss(model, batch_combined)                    
                    else:
                        # sft
                        rationale_loss = torch.tensor(-1, dtype=torch.float32)
                        answer_loss = torch.tensor(-1, dtype=torch.float32)
                        loss = model(**to_cuda(batch_, train_config.enable_fsdp, local_rank)).loss
                loss = loss / gradient_accumulation_steps
                rationale_loss = rationale_loss / gradient_accumulation_steps
                answer_loss = answer_loss / gradient_accumulation_steps
                # print("loss:", loss)
                total_loss += loss.detach().float()
                total_rationale_loss += rationale_loss.detach().float()
                total_answer_loss += answer_loss.detach().float()

                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        pbar.update(1)
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                        pbar.update(1)
                pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")
            pbar.close()
                
        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time) 

        total_loss = torch.tensor(total_loss)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_epoch_rationale_loss = total_rationale_loss / len(train_dataloader)
        train_epoch_answer_loss = total_answer_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
            train_epoch_rationale_loss = train_epoch_rationale_loss/world_size
            train_epoch_answer_loss = train_epoch_answer_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)
        
        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)
        train_rationale_loss.append(train_epoch_rationale_loss)
        train_answer_loss.append(train_epoch_answer_loss)

        
        
        # print memory used
        if train_config.enable_fsdp:
            if rank==0:
                print(f"Max CUDA memory allocated was {memtrace.peak} GB")
                print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
                print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
                print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
                print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        else:
            print(f"Max CUDA memory allocated was {memtrace.peak} GB")
            print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
            print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        
        # Update the learning rate as needed
        lr_scheduler.step()
        # if train_config.ckpt_continue
        if train_config.refine_it > 0:
            base_path = ''
            for sub_dir in train_config.ckpt_continue.split('/')[:-1]:
                base_path = os.path.join(base_path, sub_dir)
            last_dir = train_config.ckpt_continue.split('/')[-1]
            last_dir = last_dir.replace('epoch', 'refine-' + str(train_config.refine_it) + '-from-epoch')
            model_root_dir = os.path.join(base_path, last_dir)
        elif '_krsl' in train_config.dataset:
            epoch_str = train_config.model_name.split('/')[-1].split('-')[-1]
            model_root_dir = os.path.join(train_config.model_name, '..', 'load-from-epoch-' + epoch_str + '-std-lr=' + str(train_config.lr) + '-rge2=' + str(train_config.rouge2_below) + '-wd=' + str(train_config.weight_decay) + '-alpha=' + str(train_config.alpha) + '-' + train_config.dataset)
        elif 'krsl' in train_config.dataset:
            model_root_dir = os.path.join(train_config.output_dir, 'std-lr=' + str(train_config.lr) + '-rge2=' + str(train_config.rouge2_below) + '-wd=' + str(train_config.weight_decay) + '-alpha=' + str(train_config.alpha) + '-gama=' + str(train_config.gama) + '-' + train_config.dataset)
        elif 'weight' in train_config.dataset:
            model_root_dir = os.path.join(train_config.output_dir, 'std-lr=' + str(train_config.lr) + '-weight_type=' + str(train_config.weight_type) + '-step_type=' + str(train_config.step_type) + '-wd=' + str(train_config.weight_decay) + '-alpha=' + str(train_config.alpha) + '-' + train_config.dataset)
        elif 'llmst' in train_config.dataset and train_config.train_data_path != 'none':
            train_dataset_name = train_config.train_data_path.split('/')[-1].split('.')[0]
            model_root_dir = os.path.join(train_config.output_dir, 'std-lr=' + str(train_config.lr) + '-wd=' + str(train_config.weight_decay) + '-alpha=' + str(train_config.alpha) + 'train_data_name=' + train_dataset_name + '-' + train_config.dataset)
        else:
            train_dataset_name = train_config.train_data_path.split('/')[-1].split('.')[0]
            model_root_dir = os.path.join(train_config.output_dir, 'std-lr=' + str(train_config.lr) + '-wd=' + str(train_config.weight_decay) + '-alpha=' + str(train_config.alpha) + 'train_data_name=' + train_dataset_name + '-' + train_config.dataset)
        train_result_file = os.path.join(model_root_dir, 'train_result.txt')

        if True: 
            if rank == 0 and not os.path.exists(model_root_dir):
                os.mkdir(model_root_dir)
            
            if rank == 0:
                config_dict = asdict(train_cfg_ins)
                config_file_path = os.path.join(model_root_dir, 'train_config.json')
                with open(config_file_path, "w") as config_file:
                    json.dump(config_dict, config_file, indent=4)
                print(f"Train Config saved to {config_file_path}")

                config_dict = asdict(inference_cfg_ins)
                config_file_path = os.path.join(model_root_dir, 'inference_config.json')
                with open(config_file_path, "w") as config_file:
                    json.dump(config_dict, config_file, indent=4)
                print(f"Inference Config saved to {config_file_path}")

            
            if rank == 0:
                output_str = f"\nepoch={epoch + 1}, lr={lr_scheduler.get_last_lr()[0]}, train loss={train_epoch_loss}, train rationale loss={train_epoch_rationale_loss}, train answer loss={train_epoch_answer_loss}, train ppl={train_perplexity}"
                
                if krsl:
                    flag_str = 'KRSL'
                    output_str += f"Preference Learning Metrics by {flag_str}:\n"
                    for key, value in metrics.items():
                        output_str += f"\n\t\t{key}={value}"

                    output_str += "\n"

                if os.path.exists(train_result_file):
                    with open(train_result_file, 'a') as f:
                        f.write(output_str)
                else:
                    with open(train_result_file, 'w') as f:
                        f.write(output_str)


            model_dir = os.path.join(model_root_dir, 'epoch-' + str(epoch + 1))
            if rank == 0 and not os.path.exists(model_dir):
                os.mkdir(model_dir)

            if train_config.run_validation:
                eval_ppl, eval_epoch_loss = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, model_root_dir)
            else:
                eval_ppl = -1
                eval_epoch_loss = -1
            checkpoint_start_time = time.perf_counter()

            # save model
            if train_config.save_model:
                if train_config.enable_fsdp:
                    dist.barrier()

                if train_config.use_peft:
                    if train_config.enable_fsdp:
                        if rank==0:
                            print(f"we are about to save the PEFT modules")
                    else:
                        print(f"we are about to save the PEFT modules")

                    # ## saved with peft ckpt
                    if train_config.save_type == 'peft':
                        model.save_pretrained(model_dir)  

                    ## saved with merged peft ckpt
                    elif train_config.save_type == 'hf':
                        save_merged_peft_model(model, train_config.model_name, model_dir)

                    if train_config.enable_fsdp:
                        if rank==0: 
                            print(f"PEFT modules are saved in {model_dir} directory")
                    else:
                        print(f"PEFT modules are saved in {model_dir} directory")            
                else:
                    if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                        if rank != 0:
                            continue
                        # ## saved with peft ckpt
                        if train_config.save_type == 'fsdp':
                            save_model_checkpoint(
                                model, optimizer, rank, train_config, epoch=epoch+1
                            )

                        ## saved with merged peft ckpt
                        elif train_config.save_type == 'hf':
                            save_model_to_hf(model, train_config.model_name, model_dir)

                        
                    # elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                    #     print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                    #     print("=====================================================")
                        
                    #     save_model_and_optimizer_sharded(model, rank, train_config)
                    #     if train_config.save_optimizer:
                    #         save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                    #         print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                    #         print("=====================================================")

                    # if not train_config.use_peft and  train_config.save_optimizer:
                    #     save_optimizer_checkpoint(
                    #         model, optimizer, rank, train_config, epoch=epoch+1
                    #     )
                    #     print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                    #     print("=====================================================")                     
                if train_config.enable_fsdp:
                    dist.barrier()
            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.enable_fsdp:
                    if rank==0:
                        print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                else:
                    print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(best_val_loss)
            val_prep.append(eval_ppl)
            
        if train_config.enable_fsdp:
            if rank==0:
                print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, train_rationale_loss={train_epoch_rationale_loss:.4f}, train_answer_loss={train_epoch_answer_loss:.4f}, epoch time {epoch_end_time}s. \n")
        else:
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, train_rationale_loss={train_epoch_rationale_loss:.4f}, train_answer_loss={train_epoch_answer_loss:.4f}, epoch time {epoch_end_time}s. \n")


    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep) 
        avg_eval_loss = sum(val_loss)/len(val_loss) 

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    results["best_acc_epoch"] = best_acc_epoch
    # #saving the training params including fsdp setting for reference.
    # if train_config.enable_fsdp and not train_config.use_peft:
    #     save_train_params(train_config, fsdp_config, rank)
        

    if rank == 0:
        # remove peft model ckpt
        if train_config.use_peft and train_config.save_type == 'hf':
            for subdir in os.listdir(model_root_dir):
                subdir_path = os.path.join(model_root_dir, subdir)
                if os.path.isdir(subdir_path):
                    for file in os.listdir(subdir_path):
                        if not file.startswith('adapter'):
                            continue
                        file_path = os.path.join(subdir_path, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            print("remove", file_path)
    return results

def evaluation(model,train_config, eval_dataloader, local_rank, tokenizer, model_root_dir):
    # Unused
    """
    Evaluates the model on the given dataloader
    
    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions
    
    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"]) 
    model.eval()
    eval_preds = []
    eval_loss = 0.0  # Initialize evaluation loss
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch")):
            if 'user_prompt' in batch.keys() and 'original_input' in batch.keys() and 'original_output' in batch.keys():
                batch_ = {
                    'input_ids': batch['input_ids'],
                    'labels': batch['labels'],
                    'attention_mask': batch['attention_mask']
                }
            else:
                batch_ = batch
            for key in batch_.keys():
                if train_config.enable_fsdp:
                    batch_[key] = batch_[key].to(local_rank)
                else:
                    batch_[key] = batch_[key].to('cuda:0')   
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch_)
                loss = outputs.loss
                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list, 贪心策略
            preds = torch.argmax(outputs.logits, -1)
            eval_preds.extend(
                tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            )
    
    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    
    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
    eval_ppl = torch.exp(eval_epoch_loss)
    
    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank==0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")
        
    return eval_ppl, eval_epoch_loss

def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False


def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")
                
                
def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True' 
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")




def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""
    
    verify_bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    )


    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper() # llama default
    # wrapping_policy = get_mistral_wrapper() # if you use other base llm, you should modify this
    return mixed_precision_policy, wrapping_policy

def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries, 
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")

import bitsandbytes as bnb
def find_all_linear_names(model):

    #If only targeting attention blocks of the model
    target_modules = ["q_proj", "v_proj"]

    #If targeting all linear layers
    # target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']

    return target_modules
    
    
    # cls = bnb.nn.Linear4bit
    # lora_module_names = set()
    # for name, module in model.named_modules():
    #     if isinstance(module, cls):
    #         names = name.split('.')
    #         lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # return list(lora_module_names)