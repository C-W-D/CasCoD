from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="PATH/to/LLAMA/7B"
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=4
    gradient_accumulation_steps: int=1
    num_epochs: int=3
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "samsum_dataset"
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    ckpt_continue: str=None
    max_words: int=256
    alpha: float=0.5
    save_type: str='None'
    rationale_schedule: str='None'
    refine_it: int=0
    dpo_dataset: str='None'
    max_grad_norm: float=0.3
    forward_type: str='concat'
    beta: float=0.1 # no use
    dpo_loss_type: str='sigmoid'
    gama: float=0.1 # krsl loss coefficient
    krsl_alpha: float=-1.0
    krsl_beta: float=0.5
    krsl_gamma: float=0
    krsl_nll_threshold: float=0.6
    ref_model_name: str="none"
    rouge2_below: float=1.01
    n_step: int=-1
    weight_type: str="linear_decrease"
    krsl_train_data_path: str="none"
    krsl_weight_path:str="none"
    krsl_pre_dataset: str="none"
    step_type: str='sentence_split'
    train_data_path: str='none'

