import os
from pkg_resources import packaging
import fire
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    default_data_collator,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from src.configs import fsdp_config, train_config, inference_config
from src.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from src.utils import fsdp_auto_wrap_policy
from src.utils.config_utils import (
    update_config,
    generate_dataset_config,
    generate_dataset_config_by_inference,
    get_subdirectories
)
from src.utils.dataset_utils import get_preprocessed_dataset
from src.utils.dataset_utils import *
from src.inference.model_utils import load_model, load_peft_model
from src.utils.inference_utils import eval_inference
from vllm import LLM, SamplingParams
from src.model_checkpointing import load_fsdp_model_checkpoint
import gc
import torch
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

def main(**kwargs):
    update_config((inference_config, ), **kwargs)
    infer_cfg_ins = inference_config()
    update_config(infer_cfg_ins, **kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(inference_config.seed)
    torch.manual_seed(inference_config.seed)

    tokenizer = AutoTokenizer.from_pretrained(inference_config.model_name, padding_side='left', truncation_side='left')
    # tokenizer = AutoTokenizer.from_pretrained(inference_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset_config = generate_dataset_config_by_inference(inference_config, kwargs)
    print(dataset_config.dataset)
    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    print(f"--> Validation Set Length = {len(dataset_val)}")
    eval_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=inference_config.val_batch_size,
        num_workers=inference_config.num_workers_dataloader,
        pin_memory=True,
        drop_last=False,
        collate_fn=eval_dataset_collate_fn
    )

    rank_model = None
    rank_tokenizer = None
    # model = load_model(inference_config.model_name, inference_config.quantization)
    model = AutoModelForCausalLM.from_pretrained(
        inference_config.model_name,
        return_dict=True,
        load_in_8bit=inference_config.quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if inference_config.use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = model.to_bettertransformer()
            # model = BetterTransformer.transform(model)   
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
    if inference_config.load_type == 'peft':
        model_root = os.path.join(inference_config.saved_model_dir, inference_config.dataset)
        ckpt_dirs = get_subdirectories(model_root)
        for ckpt in sorted(ckpt_dirs):
            if 'epoch' not in ckpt.split('/')[-1]:
                continue
            epoch = int(ckpt.split('/')[-1].split('-')[-1])
            if epoch < inference_config.eval_epoch_begin:
                continue
            print("ckpt", ckpt)
            model = load_peft_model(model, ckpt)
            model.to('cuda')

            # if not inference_config.quantization:
            #     print("model half")
            #     model.half()  # seems to fix bugs for some users.
            model.eval()
            print("tokenizer:", tokenizer)
            print("tokenizer pad token id:", tokenizer.pad_token_id)
            macro_res = eval_inference(
                model,
                inference_config,
                eval_dataloader,
                local_rank="cuda",  
                tokenizer=tokenizer,
                model_dir=ckpt,
                train_config=None,
                infer_cfg_ins=infer_cfg_ins
            )
            print("macro_res:", macro_res)
            # break
    elif inference_config.load_type == 'fsdp':
        ckpt_dirs = get_subdirectories(inference_config.saved_model_dir)
        print(ckpt_dirs)
        for ckpt in sorted(ckpt_dirs):
            if 'epoch' not in ckpt.split('/')[-1]:
                continue
            epoch = int(ckpt.split('/')[-1].split('-')[-1].split('.')[0])
            if epoch < inference_config.eval_epoch_begin:
                continue
            print("ckpt", ckpt)
            load_fsdp_model_checkpoint(model, ckpt)
            # model.to('cuda')
            model.eval()
            print("tokenizer pad token id:", tokenizer.pad_token_id)
            macro_res = eval_inference(
                model,
                inference_config,
                eval_dataloader,
                local_rank="cuda",
                tokenizer=tokenizer,
                model_dir=ckpt,
                train_config=None,
                infer_cfg_ins=infer_cfg_ins
            )
            print("macro_res:", macro_res)
    elif inference_config.load_type == 'hf':
        if inference_config.train_dataset == 'vanilla':
            ckpt = inference_config.model_name
            available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
            print("available gpus:", available_gpus)
            model = LLM(ckpt, tensor_parallel_size=len(available_gpus))
            macro_res = eval_inference(
                model,
                inference_config,
                eval_dataloader,
                local_rank="cuda",  
                tokenizer=tokenizer,
                model_dir=ckpt,
                train_config=None,
                infer_cfg_ins=infer_cfg_ins,
                rank_model=rank_model,
                rank_tokenizer=rank_tokenizer
            )
            print("macro_res:", macro_res)      
            destroy_model_parallel()
            del model
            gc.collect()
            torch.cuda.empty_cache()
        else:
            ckpt_dirs = get_subdirectories(inference_config.saved_model_dir)
            for ckpt in sorted(ckpt_dirs):
                if 'epoch' not in ckpt.split('/')[-1]:
                    continue
                epoch = int(ckpt.split('/')[-1].split('-')[-1])
                if epoch < inference_config.eval_epoch_begin:
                    continue
                print("ckpt", ckpt)
                available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
                print("available gpus:", available_gpus)
                model = LLM(ckpt, tensor_parallel_size=len(available_gpus), gpu_memory_utilization=0.95)
                macro_res = eval_inference(
                    model,
                    inference_config,
                    eval_dataloader,
                    local_rank="cuda", 
                    tokenizer=tokenizer,
                    model_dir=ckpt,
                    train_config=None,
                    infer_cfg_ins=infer_cfg_ins,
                    rank_model=rank_model,
                    rank_tokenizer=rank_tokenizer
                )
                print("macro_res:", macro_res)
                
                destroy_model_parallel()
                del model
                gc.collect()
                torch.cuda.empty_cache()
    

if __name__ == "__main__":
    fire.Fire(main)
