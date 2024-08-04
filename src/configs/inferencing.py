from dataclasses import dataclass


@dataclass
class inference_config:
    model_name: str="PATH/to/LLAMA/7B"
    peft_model: str=None,
    quantization: bool=False,
    max_words: int=150
    max_new_tokens:int=100, #The maximum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=2.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    max_padding_length: int=32, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    num_workers_dataloader: int=1
    val_batch_size: int=1
    train_dataset: str="None"
    test_dataset: str="None"
    eval_epoch_begin: int=-1
    chat_ckpt: str="path/to/chat/ckpt"
    load_type: str='peft' # load ckpt type. peft or fsdp hf
    saved_model_dir: str='None'
    sc_cot: bool=False
    vote_num: int=1
    prompt_file: str='None'
    #  **kwargs
