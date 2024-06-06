import importlib
from functools import partial
from pathlib import Path

import torch
import sys
sys.path.append('../..')
from src.datasets import *

def load_module_from_py_file(py_file: str) -> object:
    """
    This method loads a module from a py file which is not in the Python path
    """
    module_name = Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)

    loader.exec_module(module)

    return module


def get_custom_dataset(dataset_config, tokenizer, split: str):
    if ":" in dataset_config.file:
        module_path, func_name = dataset_config.file.split(":")
    else:
        module_path, func_name = dataset_config.file, "get_custom_dataset"
        
    if not module_path.endswith(".py"):
        raise ValueError(f"Dataset file {module_path} is not a .py file.")
    
    module_path = Path(module_path)
    if not module_path.is_file():
        raise FileNotFoundError(f"Dataset py file {module_path.as_posix()} does not exist or is not a file.")
    
    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)(dataset_config, tokenizer, split)
    except AttributeError as e:
        print(f"It seems like the given method name ({func_name}) is not present in the dataset .py file ({module_path.as_posix()}).")
        raise e
    

DATASET_PREPROC = {
    "bbh_eval_dataset": partial(get_bbh_eval_dataset, max_words=150),
    "bbhtrain_eval_dataset": partial(get_bbh_eval_dataset, max_words=150),
    "bbh_dataset": partial(get_bbh_dataset, max_words=150),
    "bbh_llmst_dataset": partial(get_bbh_llmst_dataset, max_words=150),
    "bbh_llmweightst_dataset": partial(get_bbh_llmweightst_dataset, max_words=150),
    "bbh_llmstepst_dataset": partial(get_bbh_llmstepst_dataset, max_words=150),
    "bbh_llmscott_dataset": partial(get_bbh_llmscott_dataset, max_words=150),
    "bbh_llmmt_dataset": partial(get_bbh_llmmt_dataset, max_words=150),
    "bbh_llmmtre_dataset": partial(get_bbh_llmmtre_dataset, max_words=150),
    "bbh_llmmtra_dataset": partial(get_bbh_llmmtra_dataset, max_words=150),
    "bbh_llmmtcot_dataset": partial(get_bbh_llmmtcot_dataset, max_words=150),
    "bbh_llmcmt_dataset": partial(get_bbh_llmcmt_dataset, max_words=150),
    "bbh_krsl_dataset": partial(get_krsl_dataset, max_words=150),
    "bb_eval_dataset": partial(get_bb_eval_dataset, max_words=150),
    "agieval_eval_dataset": partial(get_agieval_eval_dataset, max_words=150),
    "arcc_eval_dataset": partial(get_arc_eval_dataset, max_words=150),
    "arce_eval_dataset": partial(get_arc_eval_dataset, max_words=150),
}


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )
    
    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
    )

def eval_dataset_collate_fn(batch):
    user_prompts = [item["user_prompt"] for item in batch]
    original_inputs = [item["original_input"] for item in batch]
    original_outputs = [item["original_output"] for item in batch]
    task_name = [item["task_name"] if 'task_name' in item.keys() else "None" for item in batch]
    teacher_responses = [item["teacher_response"] for item in batch]
    task_desc = [item["task_description"] for item in batch]
    collated_data = {
        "user_prompt": user_prompts,
        "original_input": original_inputs,
        "original_output": original_outputs,
        'task_name': task_name,
        'task_description': task_desc,
        'teacher_response': teacher_responses
    }

    return collated_data

def dataset_collate_fn(batch):
    user_prompts = [item["user_prompt"] for item in batch]
    original_inputs = [item["original_input"] for item in batch]
    original_outputs = [item["original_output"] for item in batch]
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]

    input_ids = torch.stack(input_ids, dim=0)
    labels = torch.stack(labels, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)

    collated_data = {
        "user_prompt": user_prompts,
        "original_input": original_inputs,
        "original_output": original_outputs,
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_masks,
    }

    return collated_data

def krsl_dataset_collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    chosens = [item["chosen"] for item in batch]
    rejecteds = [item["rejected"] for item in batch]
    chosen_input_ids = [item["chosen_input_ids"] for item in batch]
    chosen_labels = [item["chosen_labels"] for item in batch]
    chosen_attention_mask = [item["chosen_attention_mask"] for item in batch]
    chosen_weights = [item["chosen_weights"] for item in batch]
    rejected_input_ids = [item["rejected_input_ids"] for item in batch]
    rejected_labels = [item["rejected_labels"] for item in batch]
    rejected_attention_mask = [item["rejected_attention_mask"] for item in batch]
    rejected_weights = [item["rejected_weights"] for item in batch]

    chosen_input_ids = torch.stack(chosen_input_ids, dim=0)
    chosen_labels = torch.stack(chosen_labels, dim=0)
    chosen_attention_mask = torch.stack(chosen_attention_mask, dim=0)
    chosen_weights = torch.stack(chosen_weights, dim=0)
    rejected_input_ids = torch.stack(rejected_input_ids, dim=0)
    rejected_labels = torch.stack(rejected_labels, dim=0)
    rejected_attention_mask = torch.stack(rejected_attention_mask, dim=0)
    rejected_weights = torch.stack(rejected_weights, dim=0)

    collated_data = {
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
        "chosen_input_ids": chosen_input_ids,
        "chosen_labels": chosen_labels,
        "chosen_attention_mask": chosen_attention_mask,
        "chosen_weights": chosen_weights,
        "rejected_input_ids": rejected_input_ids,
        "rejected_labels": rejected_labels,
        "rejected_attention_mask": rejected_attention_mask,
        "rejected_weights": rejected_weights,
    }

    return collated_data

def llmst_dataset_collate_fn(batch):
    user_prompts = [item["user_prompt"] for item in batch]
    original_inputs = [item["original_input"] for item in batch]
    original_outputs = [item["original_output"] for item in batch]
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]


    input_ids = torch.stack(input_ids, dim=0)
    labels = torch.stack(labels, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)

    collated_data = {
        "user_prompt": user_prompts,
        "original_input": original_inputs,
        "original_output": original_outputs,
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_masks,
    }

    return collated_data

def llmweightst_dataset_collate_fn(batch):
    user_prompts = [item["user_prompt"] for item in batch]
    original_inputs = [item["original_input"] for item in batch]
    original_outputs = [item["original_output"] for item in batch]
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    weights = [item["weights"] for item in batch]


    input_ids = torch.stack(input_ids, dim=0)
    labels = torch.stack(labels, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)
    weights = torch.stack(weights, dim=0)

    collated_data = {
        "user_prompt": user_prompts,
        "original_input": original_inputs,
        "original_output": original_outputs,
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_masks,
        "weights": weights
    }

    return collated_data

def llmstepst_dataset_collate_fn(batch):
    user_prompts = [item["user_prompt"] for item in batch]
    original_inputs = [item["original_input"] for item in batch]
    original_outputs = [item["original_output"] for item in batch]
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]


    input_ids = torch.stack(input_ids, dim=0)
    labels = torch.stack(labels, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)

    collated_data = {
        "user_prompt": user_prompts,
        "original_input": original_inputs,
        "original_output": original_outputs,
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_masks,
    }

    return collated_data

def llmscott_dataset_collate_fn(batch):
    original_inputs = [item["original_input"] for item in batch]
    original_outputs = [item["original_output"] for item in batch]
    pos_input_ids = [item["pos_input_ids"] for item in batch]
    pos_labels = [item["pos_labels"] for item in batch]
    pos_attention_mask = [item["pos_attention_mask"] for item in batch]
    neg_answer_input_ids = [item["neg_answer_input_ids"] for item in batch]
    neg_answer_labels = [item["neg_answer_labels"] for item in batch]
    neg_answer_attention_mask = [item["neg_answer_attention_mask"] for item in batch]

    pos_input_ids = torch.stack(pos_input_ids, dim=0)
    pos_labels = torch.stack(pos_labels, dim=0)
    pos_attention_mask = torch.stack(pos_attention_mask, dim=0)
    neg_answer_input_ids = torch.stack(neg_answer_input_ids, dim=0)
    neg_answer_labels = torch.stack(neg_answer_labels, dim=0)
    neg_answer_attention_mask = torch.stack(neg_answer_attention_mask, dim=0)

    collated_data = {
        "original_input": original_inputs,
        "original_output": original_outputs,
        "pos_input_ids": pos_input_ids,
        "pos_labels": pos_labels,
        "pos_attention_mask": pos_attention_mask,
        "neg_answer_input_ids": neg_answer_input_ids,
        "neg_answer_labels": neg_answer_labels,
        "neg_answer_attention_mask": neg_answer_attention_mask,
    }

    return collated_data


def llmmt_dataset_collate_fn(batch):
    original_inputs = [item["original_input"] for item in batch]
    original_outputs = [item["original_output"] for item in batch]
    rationale_input_ids = [item["rationale_input_ids"] for item in batch]
    rationale_labels = [item["rationale_labels"] for item in batch]
    rationale_attention_mask = [item["rationale_attention_mask"] for item in batch]
    answer_input_ids = [item["answer_input_ids"] for item in batch]
    answer_labels = [item["answer_labels"] for item in batch]
    answer_attention_mask = [item["answer_attention_mask"] for item in batch]

    rationale_input_ids = torch.stack(rationale_input_ids, dim=0)
    rationale_labels = torch.stack(rationale_labels, dim=0)
    rationale_attention_mask = torch.stack(rationale_attention_mask, dim=0)
    answer_input_ids = torch.stack(answer_input_ids, dim=0)
    answer_labels = torch.stack(answer_labels, dim=0)
    answer_attention_mask = torch.stack(answer_attention_mask, dim=0)

    collated_data = {
        "original_input": original_inputs,
        "original_output": original_outputs,
        "rationale_input_ids": rationale_input_ids,
        "rationale_labels": rationale_labels,
        "rationale_attention_mask": rationale_attention_mask,
        "answer_input_ids": answer_input_ids,
        "answer_labels": answer_labels,
        "answer_attention_mask": answer_attention_mask,
    }

    return collated_data

def llmcmt_dataset_collate_fn(batch):
    original_inputs = [item["original_input"] for item in batch]
    original_outputs = [item["original_output"] for item in batch]
    rationale_input_ids = [item["rationale_input_ids"] for item in batch]
    rationale_labels = [item["rationale_labels"] for item in batch]
    rationale_attention_mask = [item["rationale_attention_mask"] for item in batch]
    llmrationale_answer_input_ids = [item["llmrationale_answer_input_ids"] for item in batch]
    llmrationale_answer_labels = [item["llmrationale_answer_labels"] for item in batch]
    llmrationale_answer_attention_mask = [item["llmrationale_answer_attention_mask"] for item in batch]
    selfrationale_answer_input_ids = [item["selfrationale_answer_input_ids"] for item in batch]
    selfrationale_answer_labels = [item["selfrationale_answer_labels"] for item in batch]
    selfrationale_answer_attention_mask = [item["selfrationale_answer_attention_mask"] for item in batch]

    rationale_input_ids = torch.stack(rationale_input_ids, dim=0)
    rationale_labels = torch.stack(rationale_labels, dim=0)
    rationale_attention_mask = torch.stack(rationale_attention_mask, dim=0)
    llmrationale_answer_input_ids = torch.stack(llmrationale_answer_input_ids, dim=0)
    llmrationale_answer_labels = torch.stack(llmrationale_answer_labels, dim=0)
    llmrationale_answer_attention_mask = torch.stack(llmrationale_answer_attention_mask, dim=0)

    collated_data = {
        "original_input": original_inputs,
        "original_output": original_outputs,
        "rationale_input_ids": rationale_input_ids,
        "rationale_labels": rationale_labels,
        "rationale_attention_mask": rationale_attention_mask,
        "llmrationale_answer_input_ids": llmrationale_answer_input_ids,
        "llmrationale_answer_labels": llmrationale_answer_labels,
        "llmrationale_answer_attention_mask": llmrationale_answer_attention_mask,
        "selfrationale_answer_input_ids": selfrationale_answer_input_ids,
        "selfrationale_answer_labels": selfrationale_answer_labels,
        "selfrationale_answer_attention_mask": selfrationale_answer_attention_mask
    }

    return collated_data

    # Extract fields from the batch
    # print(batch)
    original_inputs = [item["original_input"] for item in batch]
    original_outputs = [item["original_output"] for item in batch]
    pos_rationale_input_ids = [item["pos_rationale_input_ids"] for item in batch]
    pos_rationale_labels = [item["pos_rationale_labels"] for item in batch]
    pos_rationale_attention_mask = [item["pos_rationale_attention_mask"] for item in batch]
    pos_llmrationale_answer_input_ids = [item["pos_llmrationale_answer_input_ids"] for item in batch]
    pos_llmrationale_answer_labels = [item["pos_llmrationale_answer_labels"] for item in batch]
    pos_llmrationale_answer_attention_mask = [item["pos_llmrationale_answer_attention_mask"] for item in batch]
    pos_selfrationale_answer_input_ids = [item["pos_selfrationale_answer_input_ids"] for item in batch]
    pos_selfrationale_answer_labels = [item["pos_selfrationale_answer_labels"] for item in batch]
    pos_selfrationale_answer_attention_mask = [item["pos_selfrationale_answer_attention_mask"] for item in batch]
    neg_rationale_input_ids = [item["neg_rationale_input_ids"] for item in batch]
    neg_rationale_labels = [item["neg_rationale_labels"] for item in batch]
    neg_rationale_attention_mask = [item["neg_rationale_attention_mask"] for item in batch]
    neg_llmrationale_answer_input_ids = [item["neg_llmrationale_answer_input_ids"] for item in batch]
    neg_llmrationale_answer_labels = [item["neg_llmrationale_answer_labels"] for item in batch]
    neg_llmrationale_answer_attention_mask = [item["neg_llmrationale_answer_attention_mask"] for item in batch]
    neg_selfrationale_answer_input_ids = [item["neg_selfrationale_answer_input_ids"] for item in batch]
    neg_selfrationale_answer_labels = [item["neg_selfrationale_answer_labels"] for item in batch]
    neg_selfrationale_answer_attention_mask = [item["neg_selfrationale_answer_attention_mask"] for item in batch]
    chosen_input_ids = [item["chosen_input_ids"] for item in batch]
    chosen_labels = [item["chosen_labels"] for item in batch]
    chosen_attention_mask = [item["chosen_attention_mask"] for item in batch]
    rejected_input_ids = [item["rejected_input_ids"] for item in batch]
    rejected_labels = [item["rejected_labels"] for item in batch]
    rejected_attention_mask = [item["rejected_attention_mask"] for item in batch]

    # Convert tokenized inputs, labels, and attention masks to PyTorch tensors
    pos_rationale_input_ids = torch.stack(pos_rationale_input_ids, dim=0)
    pos_rationale_labels = torch.stack(pos_rationale_labels, dim=0)
    pos_rationale_attention_mask = torch.stack(pos_rationale_attention_mask, dim=0)
    pos_llmrationale_answer_input_ids = torch.stack(pos_llmrationale_answer_input_ids, dim=0)
    pos_llmrationale_answer_labels = torch.stack(pos_llmrationale_answer_labels, dim=0)
    pos_llmrationale_answer_attention_mask = torch.stack(pos_llmrationale_answer_attention_mask, dim=0)


    neg_rationale_input_ids = torch.stack(neg_rationale_input_ids, dim=0)
    neg_rationale_labels = torch.stack(neg_rationale_labels, dim=0)
    neg_rationale_attention_mask = torch.stack(neg_rationale_attention_mask, dim=0)
    neg_llmrationale_answer_input_ids = torch.stack(neg_llmrationale_answer_input_ids, dim=0)
    neg_llmrationale_answer_labels = torch.stack(neg_llmrationale_answer_labels, dim=0)
    neg_llmrationale_answer_attention_mask = torch.stack(neg_llmrationale_answer_attention_mask, dim=0)
    chosen_input_ids = torch.stack(chosen_input_ids, dim=0)
    chosen_labels = torch.stack(chosen_labels, dim=0)
    chosen_attention_mask = torch.stack(chosen_attention_mask, dim=0)
    rejected_input_ids = torch.stack(rejected_input_ids, dim=0)
    rejected_labels = torch.stack(rejected_labels, dim=0)
    rejected_attention_mask = torch.stack(rejected_attention_mask, dim=0)



    # Combine all fields into a dictionary and return
    collated_data = {
        "original_input": original_inputs,
        "original_output": original_outputs,
        "pos_rationale_input_ids": pos_rationale_input_ids,
        "pos_rationale_labels": pos_rationale_labels,
        "pos_rationale_attention_mask": pos_rationale_attention_mask,
        "pos_llmrationale_answer_input_ids": pos_llmrationale_answer_input_ids,
        "pos_llmrationale_answer_labels": pos_llmrationale_answer_labels,
        "pos_llmrationale_answer_attention_mask": pos_llmrationale_answer_attention_mask,
        "pos_selfrationale_answer_input_ids": pos_selfrationale_answer_input_ids,
        "pos_selfrationale_answer_labels": pos_selfrationale_answer_labels,
        "pos_selfrationale_answer_attention_mask": pos_selfrationale_answer_attention_mask,
        "neg_rationale_input_ids": neg_rationale_input_ids,
        "neg_rationale_labels": neg_rationale_labels,
        "neg_rationale_attention_mask": neg_rationale_attention_mask,
        "neg_llmrationale_answer_input_ids": neg_llmrationale_answer_input_ids,
        "neg_llmrationale_answer_labels": neg_llmrationale_answer_labels,
        "neg_llmrationale_answer_attention_mask": neg_llmrationale_answer_attention_mask,
        "neg_selfrationale_answer_input_ids": neg_selfrationale_answer_input_ids,
        "neg_selfrationale_answer_labels": neg_selfrationale_answer_labels,
        "neg_selfrationale_answer_attention_mask": neg_selfrationale_answer_attention_mask,
        "chosen_input_ids": chosen_input_ids,
        "chosen_labels": chosen_labels,
        "chosen_attention_mask": chosen_attention_mask,
        "rejected_input_ids": rejected_input_ids,
        "rejected_labels": rejected_labels,
        "rejected_attention_mask": rejected_attention_mask,
    }

    return collated_data