import copy
import json

import torch
from torch.utils.data import Dataset
import sys
sys.path.append('../../')
from src.datasets.utils import custom_tokenize, custom_tokenize_with_weights
from tqdm import tqdm

PROMPT_DICT = {
    "with_task_description": (
        "Task Description:\n{task_description}\nQ:{instruction}\n\nA:"
    ),
}

class EvalDataset(Dataset):
    # direct intruction fine-tuning with dataset ground truth
    def __init__(self, dataset_config, tokenizer, partition="test", max_words=30):
        self.ann = json.load(open(dataset_config.test_data_path))
        self.dataset = dataset_config.dataset
        # self.ann = self.ann[:64]
        self.max_words = dataset_config.max_words
        # tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer = tokenizer
        # self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        user_prompt = PROMPT_DICT['with_task_description'].format_map(ann)
        original_input = ann['instruction']
        original_output = ann['output']
        task_name = ann['task_name']
        teacher_response = ann['response']
        task_desc = ann['task_description']

        return {
            "user_prompt": user_prompt,
            "original_input": original_input,
            "original_output": original_output,
            'task_name': task_name,
            'task_description': task_desc,
            'teacher_response': teacher_response
        }

class InstructionDataset(Dataset):
    # direct intruction fine-tuning with dataset ground truth
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):
        if partition == "train":
            self.ann = json.load(open(dataset_config.train_data_path))
        else:
            self.ann = json.load(open(dataset_config.test_data_path))
        # self.ann = self.ann[:64]
        self.max_words = dataset_config.max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        prompt = PROMPT_DICT['with_task_description'].format_map(ann)
        user_prompt = prompt
        example = prompt + ann["output"]
        
        # assume ann with no input
        original_input = ann['instruction']
        original_output = ann['output']

        input_ids, labels, attention_masks = custom_tokenize(prompt, example, self.tokenizer, self.max_words)

        return {
            "user_prompt": user_prompt,
            "original_input": original_input,
            "original_output": original_output,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask":attention_masks
        }

class LLMSTDataset(Dataset):
    # teacher single-task dataset
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):
        if partition == "train":
            self.ann = json.load(open(dataset_config.train_data_path))
        else:
            self.ann = json.load(open(dataset_config.test_data_path))
        # self.ann = self.ann[:64]
        self.max_words = dataset_config.max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        ann['output'] = ann['response']
        prompt = PROMPT_DICT['with_task_description'].format_map(ann)
        user_prompt = prompt 
        example = prompt + ann["output"]
        
        # assume ann with no input
        original_input = ann['instruction'] 
        original_output = ann['output'] 

        input_ids, labels, attention_masks = custom_tokenize(prompt, example, self.tokenizer, self.max_words)
   
        return {
            "user_prompt": user_prompt,
            "original_input": original_input,
            "original_output": original_output,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_masks
        }

class LLMWeightSTDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):
        if partition == "train":
            self.ann = json.load(open(dataset_config.train_data_path))
        else:
            self.ann = json.load(open(dataset_config.test_data_path))
        self.max_words = dataset_config.max_words
        self.tokenizer = tokenizer
        self.weight_type = dataset_config.weight_type
        self.step_type = dataset_config.step_type

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        ann['output'] = ann['response']
        prompt = PROMPT_DICT['with_task_description'].format_map(ann)
        user_prompt = prompt
        example = prompt + ann["output"]
        
        # assume ann with no input
        original_input = ann['instruction']
        original_output = ann['output']

        input_ids, labels, attention_masks, weights = custom_tokenize_with_weights(prompt, example, self.tokenizer, self.max_words, self.weight_type, self.step_type)
   
        return {
            "user_prompt": user_prompt,
            "original_input": original_input,
            "original_output": original_output,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_masks,
            'weights': weights
        }


class LLMStepSTDataset(Dataset):
    # teacher single-task dataset but the llm response contains [stepi] 
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):
        if partition == "train":
            self.ann = json.load(open(dataset_config.train_data_path))
        else:
            self.ann = json.load(open(dataset_config.test_data_path))
        # self.ann = self.ann[:64]
        self.max_words = dataset_config.max_words
        self.tokenizer = tokenizer
        self.n_step = dataset_config.n_step

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        ann['output'] = 'Rationale: ' + ann['response']

        prompt = PROMPT_DICT['with_task_description'].format_map(ann)
        user_prompt = prompt
        example = prompt + ann["output"]
        
        # assume ann with no input
        original_input = ann['instruction']
        original_output = ann['output']

        input_ids, labels, attention_masks = custom_tokenize(prompt, example, self.tokenizer, self.max_words)
   
        return {
            "user_prompt": user_prompt,
            "original_input": original_input,
            "original_output": original_output,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_masks
        }

class LLMMTReDataset(Dataset):
    # distill step by step or MT-Re
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):
        if partition == "train":
            self.ann = json.load(open(dataset_config.train_data_path))
        else:
            self.ann = json.load(open(dataset_config.test_data_path))

        # self.ann = self.ann[:8]
        self.max_words = dataset_config.max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        original_input = ann['instruction']
        original_output = ann['output']
        
        llm_rationale = ann['response'].split('Therefore, the answer is')[0]
        llm_answer = ann['response'].split('Therefore, the answer is')[1]

        instruction = PROMPT_DICT['with_task_description'].format_map(ann)
        instruction_rationale = instruction + '\nRationale:'
        instruction_answer = instruction + '\nAnswer:'

        prompt_rationale = instruction_rationale
        example_rationale = prompt_rationale + llm_rationale

        prompt_answer = instruction_answer
        example_answer =prompt_answer + llm_answer

        rationale_input_ids, rationale_labels, rationale_attention_masks = custom_tokenize(prompt_rationale, example_rationale, self.tokenizer, self.max_words)
        answer_input_ids, answer_labels, answer_attention_masks = custom_tokenize(prompt_answer, example_answer, self.tokenizer, self.max_words)

        return {
            "original_input": original_input,
            "original_output": original_output,
            "rationale_input_ids": rationale_input_ids,
            "rationale_labels": rationale_labels,
            "rationale_attention_mask":rationale_attention_masks,
            "answer_input_ids": answer_input_ids,
            "answer_labels": answer_labels,
            "answer_attention_mask":answer_attention_masks
        }

class LLMMTRaDataset(Dataset):
    # MT-Ra
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):
        if partition == "train":
            self.ann = json.load(open(dataset_config.train_data_path))
        else:
            self.ann = json.load(open(dataset_config.test_data_path))

        # self.ann = self.ann[:8]
        self.max_words = dataset_config.max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        original_input = ann['instruction']
        original_output = ann['output']
        
        llm_rationale = ann['response'].split('Therefore, the answer is')[0]
        llm_answer = ann['response'].split('Therefore, the answer is')[1]
        llm_answer = llm_answer + '.' if not llm_answer.endswith('.') else llm_answer
        llm_rationale = 'The answer is' + llm_answer + ' Explanation: ' + llm_rationale

        instruction = PROMPT_DICT['with_task_description'].format_map(ann)
        instruction_rationale = instruction + '\n[Explanation Generation]:'
        instruction_answer = instruction + '\n[Answer Prediction]:'

        prompt_rationale = instruction_rationale
        example_rationale = prompt_rationale + llm_rationale #输出

        prompt_answer = instruction_answer
        example_answer =prompt_answer + llm_answer

        rationale_input_ids, rationale_labels, rationale_attention_masks = custom_tokenize(prompt_rationale, example_rationale, self.tokenizer, self.max_words)
        answer_input_ids, answer_labels, answer_attention_masks = custom_tokenize(prompt_answer, example_answer, self.tokenizer, self.max_words)

        return {
            "original_input": original_input,
            "original_output": original_output,
            "rationale_input_ids": rationale_input_ids,
            "rationale_labels": rationale_labels,
            "rationale_attention_mask":rationale_attention_masks,
            "answer_input_ids": answer_input_ids,
            "answer_labels": answer_labels,
            "answer_attention_mask":answer_attention_masks
        }

class LLMMTCoTDataset(Dataset):
    # MT-CoT
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):
        if partition == "train":
            self.ann = json.load(open(dataset_config.train_data_path))
        else:
            self.ann = json.load(open(dataset_config.test_data_path))
        self.max_words = dataset_config.max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        original_input = ann['instruction']
        original_output = ann['output']
        
        llm_rationale = ann['response']
        llm_answer = ann['response'].split('Therefore, the answer is')[1]
        llm_answer = llm_answer + '.' if not llm_answer.endswith('.') else llm_answer

        instruction = PROMPT_DICT['with_task_description'].format_map(ann)
        instruction_rationale = instruction + '\n[Explanation Generation]:'
        instruction_answer = instruction + '\n[Answer Prediction]:'

        prompt_rationale = instruction_rationale
        example_rationale = prompt_rationale + llm_rationale 

        prompt_answer = instruction_answer
        example_answer =prompt_answer + llm_answer

        rationale_input_ids, rationale_labels, rationale_attention_masks = custom_tokenize(prompt_rationale, example_rationale, self.tokenizer, self.max_words)
        answer_input_ids, answer_labels, answer_attention_masks = custom_tokenize(prompt_answer, example_answer, self.tokenizer, self.max_words)

        return {
            "original_input": original_input,
            "original_output": original_output,
            "rationale_input_ids": rationale_input_ids,
            "rationale_labels": rationale_labels,
            "rationale_attention_mask":rationale_attention_masks,
            "answer_input_ids": answer_input_ids,
            "answer_labels": answer_labels,
            "answer_attention_mask":answer_attention_masks
        }

class LLMMTDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):
        if partition == "train":
            self.ann = json.load(open(dataset_config.train_data_path))
        else:
            self.ann = json.load(open(dataset_config.test_data_path))

        # self.ann = self.ann[:8]
        self.max_words = dataset_config.max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        original_input = ann['instruction']
        original_output = ann['output']
        
        llm_rationale = ann['response'].split('Therefore, the answer is')[0]
        llm_answer = ann['response'].split('Therefore, the answer is')[1]

        instruction = PROMPT_DICT['with_task_description'].format_map(ann)
        instruction_rationale = instruction + '\nRationale:'
        instruction_answer = instruction + '\nAnswer:'

        prompt_rationale = instruction_rationale
        example_rationale = prompt_rationale + llm_rationale

        prompt_answer = instruction_answer
        example_answer =prompt_answer + llm_answer

        rationale_input_ids, rationale_labels, rationale_attention_masks = custom_tokenize(prompt_rationale, example_rationale, self.tokenizer, self.max_words)
        answer_input_ids, answer_labels, answer_attention_masks = custom_tokenize(prompt_answer, example_answer, self.tokenizer, self.max_words)

        return {
            "original_input": original_input,
            "original_output": original_output,
            "rationale_input_ids": rationale_input_ids,
            "rationale_labels": rationale_labels,
            "rationale_attention_mask":rationale_attention_masks,
            "answer_input_ids": answer_input_ids,
            "answer_labels": answer_labels,
            "answer_attention_mask":answer_attention_masks
        }

class LLMSCOTTDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):
        if partition == "train":
            self.ann = json.load(open(dataset_config.train_data_path))
        else:
            self.ann = json.load(open(dataset_config.test_data_path))
        self.max_words = dataset_config.max_words
        self.tokenizer = tokenizer
        self.max_length = -1


    def __len__(self):
        return len(self.ann)
    
    def getMaxLength(self):
        return self.max_length

    def __getitem__(self, index):
        ann = self.ann[index]
        data_final_pos = ann['data_final_pos']
        data_final_neg = ann['data_final_neg']

        original_input = data_final_pos['instruction']
        original_output = data_final_pos['output']

        
        pos_llm_output = data_final_pos['response']

        pos_instruction = '[Factual] ' + PROMPT_DICT['with_task_description'].format_map(data_final_pos)
        
        pos_prompt = pos_instruction
        pos_example = pos_instruction + pos_llm_output 
        pos_input_ids, pos_labels, pos_attention_masks = custom_tokenize(pos_prompt, pos_example, self.tokenizer, self.max_words)

        neg_llm_rationale = data_final_neg['response'].split('Therefore, the answer is')[0]
        neg_llm_answer = data_final_neg['response'].split('Therefore, the answer is')[1]

        neg_instruction = '[Counterfactual] ' + PROMPT_DICT['with_task_description'].format_map(data_final_neg) + neg_llm_rationale + '\nTherefore, the answer is'

        neg_prompt_llmrationale_answer = neg_instruction
        neg_example_llmrationale_answer = neg_prompt_llmrationale_answer + neg_llm_answer
        neg_llmrationale_input_ids, neg_llmrationale_labels, neg_llmrationale_attention_masks = custom_tokenize(neg_prompt_llmrationale_answer, neg_example_llmrationale_answer, self.tokenizer, self.max_words)


        return {
            "original_input": original_input,
            "original_output": original_output,
            "pos_input_ids": pos_input_ids,
            "pos_labels": pos_labels,
            "pos_attention_mask": pos_attention_masks,
            "neg_answer_input_ids": neg_llmrationale_input_ids,
            "neg_answer_labels": neg_llmrationale_labels,
            "neg_answer_attention_mask":neg_llmrationale_attention_masks,
        }


class LLMCMTDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):
        if partition == "train":
            self.ann = json.load(open(dataset_config.train_data_path))
        else:
            self.ann = json.load(open(dataset_config.test_data_path))

        # self.ann = self.ann[:256]
        self.max_words = dataset_config.max_words
        self.tokenizer = tokenizer
        self.max_length = -1


    def __len__(self):
        return len(self.ann)
    
    def getMaxLength(self):
        return self.max_length

    def __getitem__(self, index):
        ann = self.ann[index]
        original_input = ann['instruction']
        original_output = ann['output']

        
        # r, r2a
        llm_rationale = ann['response'].split('Therefore, the answer is')[0]
        llm_answer = ann['response'].split('Therefore, the answer is')[1]

        instruction = PROMPT_DICT['with_task_description'].format_map(ann)
        instruction_rationale = instruction + 'Rationale:'
        instruction_llmrationale_answer = instruction_rationale + llm_rationale + '\nTherefore, the answer is'
        instruction_selfrationale_answer = instruction_rationale + '[self-rationale]' + '\nTherefore, the answer is'
        
        prompt_rationale = instruction_rationale
        example_rationale = prompt_rationale + llm_rationale #输出

        prompt_llmrationale_answer = instruction_llmrationale_answer
        example_llmrationale_answer =prompt_llmrationale_answer + llm_answer

        prompt_selfrationale_answer = instruction_selfrationale_answer
        example_selfrationale_answer =prompt_selfrationale_answer + llm_answer
        
        rationale_input_ids, rationale_labels, rationale_attention_masks = custom_tokenize(prompt_rationale, example_rationale, self.tokenizer, self.max_words)
        llmrationale_input_ids, llmrationale_labels, llmrationale_attention_masks = custom_tokenize(prompt_llmrationale_answer, example_llmrationale_answer, self.tokenizer, self.max_words)

        return {
            "original_input": original_input,
            "original_output": original_output,
            "rationale_input_ids": rationale_input_ids,
            "rationale_labels": rationale_labels,
            "rationale_attention_mask":rationale_attention_masks,
            "llmrationale_answer_input_ids": llmrationale_input_ids,
            "llmrationale_answer_labels": llmrationale_labels,
            "llmrationale_answer_attention_mask":llmrationale_attention_masks,
            "selfrationale_answer_input_ids": prompt_selfrationale_answer,
            "selfrationale_answer_labels": example_selfrationale_answer,
            "selfrationale_answer_attention_mask": 'attention mask'
        }
    
class KRSLDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):
        self.ann = json.load(open(dataset_config.krsl_train_data_path))
        
        # self.ann = self.ann[:8]
        self.max_words = dataset_config.max_words
        self.alpha = dataset_config.krsl_alpha # > 0
        self.beta = dataset_config.krsl_beta # < 0
        self.gamma = dataset_config.krsl_gamma # > 0
        self.tokenizer = tokenizer
        self.max_length = -1
        self.rouge2_below = dataset_config.rouge2_below
        filter_data = []
        self.krsl_pre_dataset = dataset_config.krsl_pre_dataset
        try:
            import pickle
            with open(dataset_config.krsl_weight_path, 'rb') as f:
                self.weights = pickle.load(f)
            # print("load krsl weights")
            for i, ann in enumerate(tqdm(self.ann, colour='white', desc='load krsl weights')):
                if ann['rouge_2'] > self.rouge2_below:
                    continue
                chosen_operations = []
                for op in self.weights[i]['chosen_operations']:
                    if 'insert' not in op:
                        chosen_operations.append(op)
                ann['chosen_weights'] = self.assign_weights_by_operations(chosen_operations, "chosen")
                ann['chosen_operations'] = chosen_operations

                rejected_operations = []
                for op in self.weights[i]['rejected_operations']:
                    if 'insert' not in op:
                        rejected_operations.append(op)
                ann['rejected_weights'] = self.assign_weights_by_operations(rejected_operations, "rejected")
                ann['rejected_operations'] = rejected_operations
                filter_data.append(ann)
            self.ann = filter_data
                # ann['rejected_weights'] = self.assign_weights_by_operations(self.weights[i]['rejected_operations'], "rejected")
        except Exception as e:
            print(e)
            raise ValueError('Please cal the Levenshtein Distance first and save the result in a file.')
            


    def __len__(self):
        return len(self.ann)

    def assign_weights_by_operations(self, operations, s1_type):
        s1_weights = []
        for op in operations:
            if s1_type == 'chosen':
                weight = self.alpha if 'delete' in op or 'replace' in op else self.gamma
            else:  # s1_type == 'rejected'
                weight = self.beta if 'delete' in op or 'replace' in op else 0
            s1_weights.append(weight)

        s1_weights = torch.tensor(s1_weights)

        return s1_weights


    def __getitem__(self, index):
        ann = self.ann[index]
        if 'scott' in self.krsl_pre_dataset:
            prompt_str = '[Factual] ' + ann['input']
        else:
            prompt_str = ann['input']
        chosen_str = ann['chosen']
        rejected_str = ann['rejected']

        chosen_input_ids, chosen_labels, chosen_attention_mask = custom_tokenize(prompt_str, prompt_str + chosen_str, self.tokenizer, self.max_words)
        rejected_input_ids, rejected_labels, rejected_attention_mask = custom_tokenize(prompt_str, prompt_str + rejected_str, self.tokenizer, self.max_words)

        chosen_weights = ann['chosen_weights']
        rejected_weights = ann['rejected_weights']

        return {
            "prompt": prompt_str,
            "chosen": chosen_str,
            "rejected": rejected_str,
            "chosen_input_ids": chosen_input_ids,
            "chosen_labels": chosen_labels,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_weights": chosen_weights,
            "rejected_input_ids": rejected_input_ids,
            "rejected_labels": rejected_labels,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_weights": rejected_weights
        }