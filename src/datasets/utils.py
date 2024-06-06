from tqdm import tqdm
from itertools import chain
import torch
from torch.utils.data import Dataset
import copy
import re


class Concatenator(object):
    def __init__(self, chunk_size=2048):
        self.chunk_size=chunk_size
        self.residual = {"input_ids": [], "attention_mask": []}
        
    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k])) for k, v in self.residual.items()
        }

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i : i + self.chunk_size]
                    for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size) :]
                for k, v in concatenated_samples.items()
            }
        else:
            result = concatenated_samples
            self.residual = {k: [] for k in concatenated_samples.keys()}

        result["labels"] = result["input_ids"].copy()

        return result

class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size
        
        self.samples = []
        
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            }
        
        for sample in tqdm(self.dataset, desc="Preprocessing dataset"):
            buffer = {k: v + sample[k] for k,v in buffer.items()}
            
            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}
                
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def __len__(self):
        return len(self.samples)
    
def custom_tokenize(prompt, example, tokenizer, max_words):
    """
        this method encode the both args
        prompt: i.e., the input for model
        example: the gt
    """
    IGNORE_INDEX = -100
    # encode prompt
    prompt = torch.tensor(
            tokenizer.encode(prompt), dtype=torch.int64
        )
    # encode example and add <eos>
    example = tokenizer.encode(example)
    example.append(tokenizer.eos_token_id)
    example = torch.tensor(
        example, dtype=torch.int64
    )
    # right pad by zero or truncate to max words
    padding = max_words - example.shape[0]
    if padding > 0:
        example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
    elif padding < 0:
        example = example[: max_words]
    labels = copy.deepcopy(example)
    labels[: len(prompt)] = -1
    example_mask = example.ge(0)
    label_mask = labels.ge(0)
    example[~example_mask] = 0
    labels[~label_mask] = IGNORE_INDEX
    example_mask = example_mask.float()
    label_mask = label_mask.float()
    return example, labels, example_mask

def custom_tokenize_return_dict(prompt, example, tokenizer, max_words):
    """
        this method encode the both args
        prompt: i.e., the input for model
        example: the gt
    """
    IGNORE_INDEX = -100
    # encode prompt
    prompt = torch.tensor(
            tokenizer.encode(prompt), dtype=torch.int64
        )
    # encode example and add <eos>
    example = tokenizer.encode(example)
    example.append(tokenizer.eos_token_id)
    example = torch.tensor(
        example, dtype=torch.int64
    )
    # right pad by zero or truncate to max words
    padding = max_words - example.shape[0]
    if padding > 0:
        example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
    elif padding < 0:
        example = example[: max_words]
    labels = copy.deepcopy(example)
    labels[: len(prompt)] = -1
    example_mask = example.ge(0)
    label_mask = labels.ge(0)
    example[~example_mask] = 0
    labels[~label_mask] = IGNORE_INDEX
    example_mask = example_mask.float()
    label_mask = label_mask.float()
    return {
        'input_ids': example,
        'labels': labels,
        'attention_mask': example_mask
    }

def calculate_weights(n, increase=True):
    if n == 0:
        return 0, 0
    total = n * (n + 1) / 2
    if increase:
        w1 = 1 / total
        delta = w1
    else:
        w1 = n / total
        delta = - 1 / total

    return w1, delta

def count_sentences(text):
    # pattern = r'\.(\n+|;|\s+)?|;(\n+|\s+)?|\.\"(\n+|;|\s+)?|\n+'
    pattern = r'\.(\n+|;|\s+)?|\.\"(\n+|;|\s+)?|\n+'
    return len(re.findall(pattern, text))

def assign_sentence_weights(encoded_labels, tokenizer, assign_weight_type):
    # try to assign different token weight
    text = tokenizer.decode(encoded_labels)
    n_sentences = count_sentences(text)
    
    if assign_weight_type == 'linear_increase':
        w1, delta = calculate_weights(n_sentences, increase=True)
    elif assign_weight_type == 'linear_decrease':
        w1, delta = calculate_weights(n_sentences, increase=False)
    else:
        mid_point = n_sentences // 2
        w1, delta = calculate_weights(mid_point, increase=assign_weight_type == 'increase_then_decrease')

    weights = []
    current_weight = w1
    for i, token_id in enumerate(encoded_labels):
        token = tokenizer.decode([token_id])
        weights.append(current_weight)

        # if re.match(r'\.(\n+|;|\s+)?|;(\n+|\s+)?|\.\"(\n+|;|\s+)?', token):
        if re.match(r'\.(\n+|;|\s+)?|\.\"(\n+|;|\s+)?', token):
            if assign_weight_type in ['increase_then_decrease', 'decrease_then_increase'] and i >= mid_point:
                delta = -delta
            current_weight += delta

    return weights

def custom_tokenize_with_weights(prompt, example, tokenizer, max_words, assign_weight_type='linear_decrease', step_type='sentence_split'):
    IGNORE_INDEX = -100
    # encode prompt
    prompt_encoded = torch.tensor(
            tokenizer.encode(prompt), dtype=torch.int64
        )
    # encode example and add <eos>
    example_encoded = tokenizer.encode(example)
    example_encoded.append(tokenizer.eos_token_id)
    example_encoded = torch.tensor(
        example_encoded, dtype=torch.int64
    )

    # right pad by zero or truncate to max words
    padding = max_words - example_encoded.shape[0]
    if padding > 0:
        example_encoded = torch.cat((example_encoded, torch.zeros(padding, dtype=torch.int64) - 1))
    elif padding < 0:
        example_encoded = example_encoded[: max_words]

    # process index
    labels = copy.deepcopy(example_encoded)
    labels[: len(prompt_encoded)] = -1  # Set prompt part to -1

    # Mask processing
    example_mask = example_encoded.ge(0)
    label_mask = labels.ge(0)

    example_encoded[~example_mask] = 0
    labels[~label_mask] = IGNORE_INDEX

    # Convert masks to float
    example_mask = example_mask.float()
    label_mask = label_mask.float()

    # Assign weights to labels (output part only)
    output_labels = labels[len(prompt_encoded):]
    prompt_length = len(prompt_encoded)
    if step_type == 'rationale_conclusion':
        therefore_index = None
        specific_str = 'Therefore, the answer is'
        if specific_str in example:
            therefore_match = re.search(specific_str, example)
            therefore_pos = therefore_match.start()
            therefore_index = len(tokenizer.encode(example[:therefore_pos])) - 1
        weights = [0.7 if i < therefore_index else 0.3 for i in range(len(example_encoded))]
        weights_tensor = torch.tensor(weights, dtype=torch.float)
    else:
        weights = assign_sentence_weights(output_labels[output_labels != IGNORE_INDEX], tokenizer, assign_weight_type)
        weights_full = [0] * len(prompt_encoded) + weights  # Extend weights for the prompt part

        # Convert weights to tensor and adjust length
        weights_tensor = torch.tensor(weights_full, dtype=torch.float)
        if padding > 0:
            weights_tensor = torch.cat((weights_tensor, torch.zeros(padding, dtype=torch.float)))
        elif padding < 0:
            weights_tensor = weights_tensor[:max_words]

    weight_change_indices = []
    previous_weight = 0
    for i, weight in enumerate(weights_tensor):
        if weight != previous_weight and weight != 0:
            weight_change_indices.append(i)
            previous_weight = weight
    return example_encoded, labels, example_mask, weights_tensor, weight_change_indices

if __name__ == '__main__':
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoConfig,
        default_data_collator,
    )
    model_name_or_path = "./llms/llama2/llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    prompt = "Which of the following is a humorous edit of this artist or movie name: 'the godfather'?\nOptions:\n(A) the codfather\n(B) tee godfather\n(C) the godqfather\n(D) thue godfather"+ '\n\nA: '
    output = prompt + "The original name is \"the godfather\". This is the name of a classic American crime film.\n(A) \"the codfather\": Here the word \"god\" is changed to \"cod\", and this is indeed a clever and humorous play on words that ruins the original name of the movie.\n(B) \"tee godfather\": Here the word \"the\" is changed to \"tee\", but \"tee\" is not an actual word; therefore, \"tee godfather\" is not humorous.\n(C) \"the godqfather\": Here the word \"father\" is changed to \"qfather\", but \"qfather\" is not an actual word; therefore, \"the godqfather\" is not humorous.\n(D) \"thue godfather\": Here the word \"the\" is changed to \"thue\", but \"thue\" is not an actual word; therefore, \"thue godfather\" is not humorous.\nAbove the above, the only humorous edit is (A). Therefore, the answer is (A)."
    _, __, ___, ____, _____ = custom_tokenize_with_weights(prompt, output, tokenizer, 512, 'decrease_then_increase', 'rationale_conclusion')