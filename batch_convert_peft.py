# 将一个目录下所有权重从adapter转为bin形式

import fire
import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from src.model_checkpointing import load_peft_model_then_save
import os
import time

def main(base_model_dir, peft_model_dir):
    # base_model_dir: contain checkpoint of base model 
    # peft_model_dir: contain peft ckpt, e.g. adapter_model.bin

    # load and save
    for subdir in os.listdir(peft_model_dir):
        subdir_path = os.path.join(peft_model_dir, subdir)
        if 'runs' in subdir_path:
            continue
        if os.path.isdir(subdir_path) and subdir.startswith("epoch"):
            load_peft_model_then_save(subdir_path, base_model_dir, subdir_path)
            
    time.sleep(5)
    # delete adapter
    for subdir in os.listdir(peft_model_dir):
        subdir_path = os.path.join(peft_model_dir, subdir)
        if 'runs' in subdir_path:
            continue
        if os.path.isdir(subdir_path) and subdir.startswith("epoch"):
            for file in os.listdir(subdir_path):
                if not file.startswith('adapter'):
                    continue
                file_path = os.path.join(subdir_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print("remove", file_path)


if __name__ == "__main__":
    fire.Fire(main)
    # base_model = '/llama2/llama-2-7b-hf'
    # peft_model = ''

    # main(base_model_dir=base_model,
    #      peft_model_dir=peft_model)