# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import inspect
from dataclasses import asdict
from peft import (
    LoraConfig,
    AdaptionPromptConfig,
    PrefixTuningConfig,
)
import sys
sys.path.append('../..')
from src.configs import datasets, lora_config, llama_adapter_config, prefix_config, train_config
from src.utils.dataset_utils import DATASET_PREPROC


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warm user
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, train_config):
                print(f"Warning: unknown parameter {k}")
                        
                        
def generate_peft_config(train_config, kwargs):
    configs = (lora_config, llama_adapter_config, prefix_config)
    peft_configs = (LoraConfig, AdaptionPromptConfig, PrefixTuningConfig)
    names = tuple(c.__name__.rstrip("_config") for c in configs)
    
    assert train_config.peft_method in names, f"Peft config not found: {train_config.peft_method}"
    
    config = configs[names.index(train_config.peft_method)]()
    
    update_config(config, **kwargs)
    params = asdict(config)
    peft_config = peft_configs[names.index(train_config.peft_method)](**params)
    
    return peft_config


def generate_dataset_config(train_config, kwargs):
    names = tuple(DATASET_PREPROC.keys())
        
    dataset_config = {k:v for k, v in inspect.getmembers(datasets)}[train_config.dataset]()
        
    dataset_name = dataset_config.dataset
    update_config(dataset_config, **kwargs)
    dataset_config.dataset = dataset_name
    dataset_config.max_words = train_config.max_words
    if 'krsl' in dataset_name:
        dataset_config.krsl_alpha = train_config.krsl_alpha
        dataset_config.krsl_beta = train_config.krsl_beta
        dataset_config.krsl_gamma = train_config.krsl_gamma
        dataset_config.rouge2_below = train_config.rouge2_below
        dataset_config.krsl_pre_dataset = train_config.krsl_pre_dataset
        dataset_config.krsl_train_data_path = train_config.krsl_train_data_path
        dataset_config.krsl_weight_path = train_config.krsl_weight_path
    if 'step' in dataset_name:
        dataset_config.n_step = train_config.n_step
        if 'llmst' in dataset_name and train_config.n_step == 2:
            pass
        else:
            dataset_config.train_data_path = dataset_config.train_data_path.replace('num', str(train_config.n_step))
    if 'weight' in dataset_name:
        dataset_config.weight_type = train_config.weight_type
        dataset_config.step_type = train_config.step_type
    if train_config.train_data_path != 'none':
        dataset_config.train_data_path = train_config.train_data_path

    return  dataset_config

def generate_dataset_config_by_inference(inference_config, kwargs):
    names = tuple(DATASET_PREPROC.keys())

    if 'bbhtrain' in inference_config.test_dataset:
        dataset_name = 'bbhtrain_eval_dataset'
    elif 'bbh' in inference_config.test_dataset:
        dataset_name = 'bbh_eval_dataset'
    elif 'bb' in inference_config.test_dataset:
        dataset_name = 'bb_eval_dataset'
    elif 'agieval' in inference_config.test_dataset:
        dataset_name = 'agieval_eval_dataset'
    elif 'arcc' in inference_config.test_dataset:
        dataset_name = 'arcc_eval_dataset'
    elif 'arce' in inference_config.test_dataset:
        dataset_name = 'arce_eval_dataset'
    else:
        dataset_name = 'none'
        
    # assert train_config.dataset in names, f"Unknown dataset: {inference_config.dataset}"
    
    dataset_config = {k:v for k, v in inspect.getmembers(datasets)}[dataset_name]()
        
    dataset_name = dataset_config.dataset
    update_config(inference_config, **kwargs)
    dataset_config.dataset = dataset_name
    
    dataset_config.max_words = inference_config.max_words

    return  dataset_config

def get_subdirectories(directory):
    subdirectories = []
    import os
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) or item_path.endswith('.pt'):
            subdirectories.append(item_path)
    return subdirectories