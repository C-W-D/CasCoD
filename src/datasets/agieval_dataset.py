import copy
import json

import torch
from torch.utils.data import Dataset
import sys
sys.path.append('../../')
from src.datasets.utils import custom_tokenize

PROMPT_DICT = {
    "prompt": (
        "{instruction}\n\nA:"
    ),
}

class EvalDataset(Dataset):
    # direct intruction fine-tuning with dataset ground truth
    def __init__(self, dataset_config, tokenizer, partition="test", max_words=30):
        self.ann = json.load(open(dataset_config.test_data_path))
        self.dataset = dataset_config.dataset
        # self.ann = self.ann[:8]
        self.max_words = dataset_config.max_words
        # tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer = tokenizer
        # self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        user_prompt = PROMPT_DICT['prompt'].format_map(ann)
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

