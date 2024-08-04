import os
import time
import yaml
from pathlib import Path
from pkg_resources import packaging
import json
import copy
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import AutoTokenizer
from dataclasses import dataclass, asdict
import sys
sys.path.append('../..')
from src.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint
from src.policies import fpSixteen,bfSixteen_mixed, get_llama_wrapper
from src.utils.memory_utils import MemoryTrace
from src.inference.safety_utils import get_safety_checker
from src.inference.model_utils import load_model, load_peft_model
from src.utils.metric_utils import compute_metrics
from vllm import LLM, SamplingParams

from collections import Counter

def find_majority_answer_text(text_list):
    def extract_answers(text_list):
        answers = []
        for text in text_list:
            parts = text.split('Therefore, the answer is', 1)
            if len(parts) > 1:
                answer = parts[1].strip()
                answers.append(answer)
        return answers

    def majority_vote(answers):
        if not answers:
            return None
        counter = Counter(answers)
        most_common_answer = counter.most_common(1)[0][0]
        return most_common_answer

    def get_full_text_for_answer(text_list, answer):
        for text in text_list:
            if answer in text:
                return text
        return None

    try:
        answers = extract_answers(text_list)

        most_common_answer = majority_vote(answers)

        result_text = get_full_text_for_answer(text_list, most_common_answer)
        if result_text is None:
            result_text = text_list[0]
    except:
        result_text = text_list[0]

    return result_text

def eval_inference(model, inference_config, eval_dataloader, local_rank, tokenizer, model_dir, train_config=None, infer_cfg_ins=None, rank_model=None, rank_tokenizer=None):
    ### only support single gpu 
    """
    evaluation + inference
    advanced by train_utils.evaluation

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions
    
    Returns record_result(...)
    """
    # if train_config.enable_fsdp:
    #     world_size = int(os.environ["WORLD_SIZE"]) 
    if inference_config.load_type != 'hf':
        model.eval()
    eval_preds = []
    model_inputs = []
    original_inputs = []
    original_outputs = []
    teacher_responses = []
    task_descs = []
    task_names = []
    sample_para = None
    stop_tokens = ["---", "```output"]
    if inference_config.sc_cot:
        sample_para = SamplingParams(
            temperature=inference_config.temperature,
            top_p=inference_config.top_p,
            top_k=inference_config.top_k,
            max_tokens=inference_config.max_new_tokens,
            n=1,
            stop=stop_tokens
        )
    else:
        if inference_config.do_sample is True:
            sample_para = SamplingParams(
                temperature=inference_config.temperature,
                top_p=inference_config.top_p,
                top_k=inference_config.top_k,
                max_tokens=inference_config.max_new_tokens,
                n=1,
                stop=stop_tokens
            )
        else:
            sample_para = SamplingParams(
                temperature=0,
                top_p=1.0,
                top_k=-1,
                max_tokens=inference_config.max_new_tokens,
                n=1,
                stop=stop_tokens
            )
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch")):
            # if 'user_prompt' in batch.keys() and 'original_input' in batch.keys() and 'original_output' in batch.keys():
            original_inputs.extend(batch['original_input'])
            original_outputs.extend(batch['original_output'])
            model_inputs.extend(batch['user_prompt'])
            task_descs.extend(batch['task_description'])
            user_prompts = batch['user_prompt']
            if 'task_name' in batch.keys():
                task_names.extend(batch['task_name'])
            else:
                task_names.extend(['None' for cnt in len(batch['original_input'])])
            teacher_responses.extend(batch['teacher_response'])

            # print('original user prompt:', user_prompts)
            if 'scott' in inference_config.saved_model_dir:
                print('scott')
                ## same with train data
                user_prompts = ['[Factual] ' + x for x in user_prompts]
            elif 'llmmtcot' in inference_config.saved_model_dir:
                print('llmmtcot')
                ## same with train data
                user_prompts = [x + '\n[Answer Prediction]: ' for x in user_prompts]
            elif 'llmmtra' in inference_config.saved_model_dir:
                print('llmmtra')
                ## same with train data
                user_prompts = [x + '\n[Answer Prediction]: ' for x in user_prompts]
            elif 'llmmtre' in inference_config.saved_model_dir:
                print('llmmtre')
                user_prompts = [x + '\nAnswer: ' for x in user_prompts]
            elif 'llmcmt' in inference_config.saved_model_dir:
                print('llmcmt')
                if inference_config.load_type != 'hf':
                    ## inference with step-by-step, first generate rationle, then output answer followed by rationale
                    selfrationale_prompts = [x + '\nRationale:' for x in user_prompts]
                    batch_r = tokenizer(selfrationale_prompts, padding='max_length', truncation=True, max_length=inference_config.max_padding_length, return_tensors="pt")
                    for key in batch_r.keys():
                        batch_r[key] = batch_r[key].to(local_rank)
                    with torch.no_grad():
                        outputs_r = model.generate(
                            **batch_r,
                            max_new_tokens=inference_config.max_new_tokens,
                            do_sample=inference_config.do_sample,
                            top_p=inference_config.top_p,
                            temperature=inference_config.temperature,
                            min_length=inference_config.min_length,
                            use_cache=inference_config.use_cache,
                            top_k=inference_config.top_k,
                            repetition_penalty=inference_config.repetition_penalty,
                            length_penalty=inference_config.length_penalty
                        )
                    selfrationales = tokenizer.batch_decode(outputs_r, skip_special_tokens=True)
                    user_prompts = [x + '\nTherefore, the answer is' for x in selfrationales]
                else:
                    print('use vllm')
                    ### use vllm for generation
                    stop_tokens = ["</s>", "---", "```output"]
                    selfrationale_prompts = [x + '\nRationale:' for x in user_prompts]
                    outputs = model.generate(selfrationale_prompts, sample_para)
                    selfrationales = [output.outputs[0].text for output in outputs]
                    user_prompts = [x + y + '\nTherefore, the answer is' for x, y in zip(selfrationale_prompts, selfrationales)]
            elif 'vanilla' in inference_config.saved_model_dir:
                prompt_prefix = ''
                print("pf:", inference_config.prompt_file)
                with open(inference_config.prompt_file, 'r') as f:
                    prompt_prefix = f.read()
                if 'cot' not in inference_config.prompt_file:
                    user_prompts = [prompt_prefix + '\n' + x for x in user_prompts]
                else:
                    user_prompts = [prompt_prefix + '\n' + x + ' Let\'s think step by step.' for x in user_prompts]
            else:
                pass

            print("inputs", user_prompts[:2])
            if inference_config.load_type != 'hf':
                batch_ = tokenizer(user_prompts, padding='max_length', truncation=True, max_length=inference_config.max_padding_length, return_tensors="pt")
                for key in batch_.keys():
                    batch_[key] = batch_[key].to(local_rank)

                with torch.no_grad():
                    outputs = model.generate(
                        **batch_,
                        max_new_tokens=inference_config.max_new_tokens,
                        do_sample=inference_config.do_sample,
                        top_p=inference_config.top_p,
                        temperature=inference_config.temperature,
                        min_length=inference_config.min_length,
                        use_cache=inference_config.use_cache,
                        top_k=inference_config.top_k,
                        repetition_penalty=inference_config.repetition_penalty,
                        length_penalty=inference_config.length_penalty
                    )
                responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                eval_preds.extend(
                    responses
                )
            else:
                new_user_prompts = []
                for p in user_prompts:
                    for _ in range(inference_config.vote_num):
                        new_user_prompts.append(p)
                outputs = model.generate(new_user_prompts, sample_para)
                if inference_config.train_dataset == 'vanilla':
                    responses = []
                    for idx, user_prompt in enumerate(user_prompts):
                        s, e = idx * inference_config.vote_num, (idx + 1) * inference_config.vote_num
                        batch_output_texts = []
                        for oidx in range(s, e):
                            batch_output_texts.append(outputs[oidx].outputs[0].text)
                        responses.append(find_majority_answer_text(batch_output_texts))
                    # responses = [output.outputs[0].text for user_prompt, output in zip(user_prompts, outputs)]
                else:
                    responses = [user_prompt + output.outputs[0].text for user_prompt, output in zip(user_prompts, outputs)]
                eval_preds.extend(
                    responses
                )
            print("response:", responses[:2])

    # complete the orignal output with detail choice contents
    for i, x in enumerate(original_outputs):
        if '(' in x and ')' in x and x in original_inputs[i]:
            original_outputs[i] = x + original_inputs[i].split(x)[1].split('\n')[0]


    return record_result(original_inputs, original_outputs, task_descs, task_names, teacher_responses, model_inputs, eval_preds, model_dir, inference_config, infer_cfg_ins, rank_model, rank_tokenizer)


def record_result(original_inputs, original_outputs, task_descs, task_names, teacher_responses, model_inputs, eval_preds, model_dir, inference_config, infer_cfg_ins, rank_model, rank_tokenizer):
    model_outputs = copy.deepcopy(eval_preds)
    macro_res, details = compute_metrics(rank_model=rank_model, rank_tokenizer=rank_tokenizer, task_descs=task_descs, task_names=task_names, teacher_responses=teacher_responses, model_inputs=model_inputs, original_inputs=original_inputs, original_outputs=original_outputs, model_outputs=model_outputs, eval_preds=eval_preds, inference_config=inference_config)
    
    micro_res = [
        {
            'description': 'micro test result',
            'details': details
        },
    ]
    all_res = [macro_res, micro_res]
    name = model_dir.split('/')[-1].split('-')[-1]
    if name.endswith('.pt'):
        epoch = int(name.split('.')[0])
    else:
        try:
            epoch = int(name)
        except Exception as e:
            epoch = 0
    if os.path.isdir(model_dir):
        result_dir = os.path.join(model_dir, '../results')
    else:
        result_dir = os.path.join(os.path.dirname(model_dir), '../results')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    ## if train dataset != test dataset
    if inference_config.train_dataset != inference_config.test_dataset:
        result_dir = os.path.join(result_dir, inference_config.test_dataset)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
    if inference_config.train_dataset == 'vanilla':
        result_file = os.path.join(result_dir, inference_config.prompt_file.split('/')[-1].split('.')[0] + '-eval_result.json')
    else:
        result_file = os.path.join(result_dir, 'epoch-' + str(epoch) + '-eval_result.json')
    with open(result_file, 'w', encoding='utf-8') as f:
      json.dump(all_res, f, ensure_ascii=False, indent=4)
    print(f'Write evaluation result to {result_file} successfully.')

    config_dict = asdict(infer_cfg_ins)
    config_file_path = os.path.join(result_dir, 'inference_config.json')
    with open(config_file_path, "w") as config_file:
        json.dump(config_dict, config_file, indent=4)
    print(f"Config saved to {config_file_path}")
    return macro_res
    


def single_inference(inference_config):
    # Unused
    """
    Single inference
    """
    
    # prompt template + instruction + input 
    if inference_config.prompt_file is not None:
        assert os.path.exists(
            inference_config.prompt_file
        ), f"Provided Prompt file does not exist {inference_config.prompt_file}"
        with open(inference_config.prompt_file, "r") as f:
            user_prompt = "\n".join(f.readlines())
    elif not sys.stdin.isatty():
        user_prompt = "\n".join(sys.stdin.readlines())
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)
    
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(inference_config.seed)
    torch.manual_seed(inference_config.seed)
    
    model = load_model(inference_config.model_name, inference_config.quantization)
    if inference_config.peft_model:
        model = load_peft_model(model, inference_config.peft_model)

    model.eval()
    
    if inference_config.use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)    
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = AutoTokenizer.from_pretrained(inference_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model.resize_token_embeddings(model.config.vocab_size + 1) 
    
    safety_checker = get_safety_checker(inference_config.enable_azure_content_safety,
                                        inference_config.enable_sensitive_topics,
                                        inference_config.enable_salesforce_content_safety,
                                        )

    # Safety check of the user prompt
    safety_results = [check(user_prompt) for check in safety_checker]
    are_safe = all([r[1] for r in safety_results])
    if are_safe:
        print("User prompt deemed safe.")
        print(f"User prompt:\n{user_prompt}")
    else:
        print("User prompt deemed unsafe.")
        for method, is_safe, report in safety_results:
            if not is_safe:
                print(method)
                print(report)
        print("Skipping the inference as the prompt is not safe.")
        sys.exit(1)  # Exit the program with an error status
        
    def inference_(model, user_prompt, tokenizer, inference_config):
        """
        inference the model with the user prompt
        Args:
            model: the loaded model
            user_prompt: user's prompt or input
            tokenizer: the model's tokenizer
        Returns:
            output_text: the response of model for the user prompt
        """
        # tokenize
        batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=inference_config.max_padding_length, return_tensors="pt")

        batch = {k: v.to("cuda") for k, v in batch.items()}
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=inference_config.max_new_tokens,
                do_sample=inference_config.do_sample,
                top_p=inference_config.top_p,
                temperature=inference_config.temperature,
                min_length=inference_config.min_length,
                use_cache=inference_config.use_cache,
                top_k=inference_config.top_k,
                repetition_penalty=inference_config.repetition_penalty,
                length_penalty=inference_config.length_penalty,
                **inference_config.kwargs 
            )
        e2e_inference_time = (time.perf_counter()-start)*1000
        print(f"the inference time is {e2e_inference_time} ms")
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text

    return inference_(model, user_prompt, tokenizer, inference_config)
    
    # # Safety check of the model output
    # safety_results = [check(output_text) for check in safety_checker]
    # are_safe = all([r[1] for r in safety_results])
    # if are_safe:
    #     # print("User input and model output deemed safe.")
    #     # print(f"Model output:\n{output_text}")
    #     return output_text
    # else:
    #     print("Model output deemed unsafe.")
    #     for method, is_safe, report in safety_results:
    #         if not is_safe:
    #             print(method)
    #             print(report)
    #     return 'unsafe'