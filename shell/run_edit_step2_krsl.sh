#!/bin/bash
cd '../'
lr=5e-6 # base lr
weight_decay=0 # for adamW
gamma=0.95
alpha=0.5 # alpha * answer loss + (1-alpha) * rloss
gama=0.1 # alpha * answer loss + (1-alpha) * rloss + gama * preference loss, no use
krsl_alpha=1.0
krsl_beta=-0.025
krsl_gamma=0.0
# following two paras are fixed for edit.
krsl_nll_threshold=20 # lower is stricter, higher is looser. 20 denotes this para almost doesnt play a role
rouge2_below=1.01 # 0-1. 1.01 denotes filtering is disabled.

dataset="bbh_krsl_dataset"
krsl_pre_dataset="bbh_llmst_dataset"
krsl_train_data_path="./dataset/bbh/bbh_all_data/all_task_train_right_preference_with_answer.json"
krsl_weight_path="./dataset/bbh/bbh_all_data/all_task_train_right_preference_with_answer_precal.pkl"
forward_type='concat'
beta=0.1 # for dpo, no use
dpo_loss_type='sigmoid' # for dpo, no use

model_name="path/to/llmst-tuned-model/epoch-x" # path to llama 7b
last_dirname="${model_name##*/}"
refine_it=0 
save_type="hf"
output_dir="../slm/$save_type/$last_dirname"
# mkdir -p "$output_dir"
ckpt_continue=None
max_words=1024 
num_epochs=10
batch_size_training=16
gradient_accumulation_steps=4
num_workers_dataloader=1

use_peft=True
peft_method="lora" # None , llama_adapter, prefix
seed=42 # seed value for reproducibility
enable_fsdp=True
low_cpu_fsdp=False
run_validation=False
use_fp16=False #use_fp16 boolean flag to specify using FP16 for mixed precision, defatults to False. We recommond not setting this flag, and only set mixed_precision that will use BF16, this will help with speed and memory savings while avoiding challenges of scaler accuracies with FP16.
mixed_precision=True 
val_batch_size=1
freeze_layers=False
num_freeze_layers=1
quantization=False 
one_gpu=False
save_model=True
dist_checkpoint_root_folder="./output/fsdp/$last_dirname" # will be used if using FSDP
# mkdir -p "$dist_checkpoint_root_folder"
# mkdir -p "$dist_checkpoint_root_folder/$dataset"
dist_checkpoint_folder="fine-tuned" # will be used if using FSDP
save_optimizer=False # will be used if using FSDP
use_fast_kernels=False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
# mixed_precision=True
fsdp_activation_checkpointing=True
pure_bf16=False
optimizer="AdamW"

# inference config, no use
max_new_tokens=512  # The maximum numbers of tokens to generate
prompt_file=None
do_sample=True  # Whether or not to use sampling ; use greedy decoding otherwise.
min_length=None  # The minimum length of the sequence to be generated, input prompt + min_new_tokens
top_p=1.0  # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
temperature=1.0  # [optional] The value used to modulate the next token probabilities.
top_k=50  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
repetition_penalty=1.0  # The parameter for repetition penalty. 1.0 means no penalty.
length_penalty=1  # [optional] Exponential penalty to the length that is used with beam-based generation.
max_padding_length=None  # the max padding length to be used with tokenizer padding the prompts.
peft_model=None
quantization=False
enable_azure_content_safety=False  # Enable safety check with Azure content safety api
enable_sensitive_topics=False  # Enable check for sensitive topics using AuditNLG APIs
enable_salesforce_content_safety=False  # Enable safety check with Salesforce safety flan t5
use_fast_kernels=False  # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
use_cache=True  # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.

mkdir -p ./log
dataset_dir="./log/$dataset"
mkdir -p "$dataset_dir"
mkdir -p "$dataset_dir/$last_dirname"
log_file="$dataset_dir/$last_dirname/log.txt"

export CUDA_VISIBLE_DEVICES="0,1,2,3"
torchrun --nnodes 1 --nproc_per_node 4 --master-port 29828 ./finetuning.py \
    --max_words $max_words \
    --enable_fsdp $enable_fsdp \
    --model_name "$model_name" \
    --low_cpu_fsdp $low_cpu_fsdp \
    --run_validation $run_validation \
    --batch_size_training $batch_size_training \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_epochs $num_epochs \
    --num_workers_dataloader $num_workers_dataloader \
    --lr $lr \
    --weight_decay $weight_decay \
    --gamma $gamma \
    --seed $seed \
    --use_fp16 $use_fp16 \
    --mixed_precision $mixed_precision \
    --val_batch_size $val_batch_size \
    --dataset "$dataset" \
    --peft_method "$peft_method" \
    --use_peft $use_peft \
    --output_dir "$output_dir" \
    --freeze_layers $freeze_layers \
    --num_freeze_layers $num_freeze_layers \
    --quantization $quantization \
    --one_gpu $one_gpu \
    --save_model $save_model \
    --dist_checkpoint_root_folder "$dist_checkpoint_root_folder" \
    --dist_checkpoint_folder "$dist_checkpoint_folder" \
    --save_optimizer $save_optimizer \
    --use_fast_kernels $use_fast_kernels \
    --mixed_precision $mixed_precision \
    --fsdp_activation_checkpointing $fsdp_activation_checkpointing \
    --pure_bf16 $pure_bf16 \
    --optimizer "$optimizer" \
    --ckpt_continue $ckpt_continue \
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
    --save_type $save_type \
    --alpha $alpha \
    --refine_it $refine_it \
    --forward_type "$forward_type" \
    --beta $beta \
    --gama $gama \
    --krsl_alpha $krsl_alpha \
    --krsl_beta $krsl_beta \
    --krsl_gamma $krsl_gamma \
    --krsl_nll_threshold $krsl_nll_threshold \
    --rouge2_below $rouge2_below \
    --krsl_pre_dataset $krsl_pre_dataset \
    --krsl_train_data_path $krsl_train_data_path \
    --krsl_weight_path $krsl_weight_path \
    --max_padding_length $max_padding_length 2>&1 | tee "$log_file"