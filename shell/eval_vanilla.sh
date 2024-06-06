#!/bin/bash
cd ../
model_name="./dcw/dcw_home/slm/hf/llama-2-7b-hf/vanilla/epoch-0" # path to llama 7b
eval_epoch_begin=-1
num_workers_dataloader=1
val_batch_size=256
train_dataset='vanilla' 
test_dataset="bb_eval_dataset"
load_type='hf'
last_dirname="${model_name##*/}"
saved_model_dir="./output/$load_type/$last_dirname/$train_dataset"
prompt_file="./prompt_for_vanilla/prompt_3shotcot_for_vanilla.txt"
mkdir -p "$saved_model_dir"
quantization=False
max_words=150 # input sequence max words
max_new_tokens=1024  # The maximum numbers of tokens to generate

seed=42  # seed value for reproducibility
do_sample=True  # Whether or not to use sampling ; use greedy decoding otherwise.
min_length=None  # The minimum length of the sequence to be generated, input prompt + min_new_tokens
use_cache=True  # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
top_p=1.0  # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
temperature=1.0  # [optional] The value used to modulate the next token probabilities.
top_k=50  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
repetition_penalty=1.0  # The parameter for repetition penalty. 1.0 means no penalty.
length_penalty=1.0  # [optional] Exponential penalty to the length that is used with beam-based generation.
enable_azure_content_safety=False  # Enable safety check with Azure content safety api
enable_sensitive_topics=False  # Enable check for sensitive topics using AuditNLG APIs
enable_salesforce_content_safety=False  # Enable safety check with Salesforce safety flan t5
max_padding_length=256  # the max padding length to be used with tokenizer padding the prompts.
use_fast_kernels=False  # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels

export CUDA_VISIBLE_DEVICES="2"
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="56664"
python evaluation.py \
    --model_name "$model_name" \
    --num_workers_dataloader $num_workers_dataloader \
    --seed $seed \
    --val_batch_size $val_batch_size \
    --max_words $max_words \
    --train_dataset "$train_dataset" \
    --test_dataset "$test_dataset" \
    --peft_model "$peft_model" \
    --max_new_tokens $max_new_tokens \
    --prompt_file "$prompt_file" \
    --do_sample $do_sample \
    --min_length $min_length \
    --use_cache $use_cache \
    --top_p $top_p \
    --temperature $temperature \
    --top_k $top_k \
    --repetition_penalty $repetition_penalty \
    --length_penalty $length_penalty \
    --enable_azure_content_safety $enable_azure_content_safety \
    --enable_sensitive_topics $enable_sensitive_topics \
    --enable_salesforce_content_safety $enable_salesforce_content_safety \
    --max_padding_length $max_padding_length \
    --use_fast_kernels $use_fast_kernels \
    --load_type $load_type \
    --eval_epoch_begin $eval_epoch_begin \
    --saved_model_dir $saved_model_dir