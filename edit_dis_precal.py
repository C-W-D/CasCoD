from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    default_data_collator,
    TrainingArguments,
    BitsAndBytesConfig
)
import json
import torch
from tqdm import tqdm
from src.datasets.utils import custom_tokenize
import pickle

def levenshtein_operations_tokens(s1_tokens, s2_tokens):
    # Process prefix for optimizing time cost
    prefix_len = 0
    while prefix_len < len(s1_tokens) and prefix_len < len(s2_tokens) and s1_tokens[prefix_len] == s2_tokens[prefix_len]:
        prefix_len += 1
    # Remove the prefix
    s1_tokens = s1_tokens[prefix_len:]
    s2_tokens = s2_tokens[prefix_len:]

    # Minimum operations s1 should perform compared to s2
    rows = len(s1_tokens) + 1
    cols = len(s2_tokens) + 1
    dp = [[0 for _ in range(cols)] for _ in range(rows)]

    # Initialize the dynamic programming matrix
    for i in range(1, rows):
        dp[i][0] = i
    for i in range(1, cols):
        dp[0][i] = i

    # Calculate Levenshtein distance and fill the matrix
    for col in range(1, cols):
        for row in range(1, rows):
            cost = 0 if s1_tokens[row - 1] == s2_tokens[col - 1] else 1
            dp[row][col] = min(dp[row - 1][col] + 1,      # Deletion
                            dp[row][col - 1] + 1,      # Insertion
                            dp[row - 1][col - 1] + cost) # Replacement

    # Backtrace to find the operations
    operations = []
    row, col = rows - 1, cols - 1
    while row > 0 or col > 0:
        if row > 0 and dp[row][col] == dp[row - 1][col] + 1:
            operations.append(f'delete {s1_tokens[row - 1]}')
            row -= 1
        elif col > 0 and dp[row][col] == dp[row][col - 1] + 1:
            operations.append(f'insert {s2_tokens[col - 1]}')
            col -= 1
        else:
            operation = 'none' if s1_tokens[row - 1] == s2_tokens[col - 1] else f'replace {s1_tokens[row - 1]} with {s2_tokens[col - 1]}'
            operations.append(operation)
            row -= 1
            col -= 1

    # Add none operations for the prefix
    operations.extend(['none'] * prefix_len)
    return operations[::-1], dp

def assign_weights(s1_tokens, s2_tokens, s1_type):
    """
    For each sample, calculate the weights assigned to the minimum edit operations of s1 compared to s2.
    Args:
        s1_tokens, s2_tokens: token sequences of batch, each batch with shape [batch_size, seq_len]
        s1_type: 'chosen' or 'rejected' indicating the type of s1 sequence
        alpha, beta, gamma: weight parameters
    Return:
        s1_weights: weight tensor with shape [batch_size, seq_len]
    """
    seq_len = s1_tokens.shape[0]
    operations, _ = levenshtein_operations_tokens(s1_tokens, s2_tokens)

    s1_weights = []
    for op in operations:
        if s1_type == 'chosen':
            # Conversely, reverse the semantics in the operations
            weight = -1.0 if 'delete' in op or 'replace' in op else 0
        else:  # s1_type == 'rejected'
            weight = 0.5 if 'delete' in op or 'replace' in op else 0
        s1_weights.append(weight)

    # Pad to seq_len length
    if len(s1_weights) < seq_len:
        s1_weights += [0] * (seq_len - len(s1_weights))
    else:
        s1_weights = s1_weights[:seq_len]
    s1_weights = torch.tensor(s1_weights)

    return s1_weights, operations

def main():
    data_file = './dataset/bbh/bbh_all_data/all_task_train_preference_with_answer.json'
    model_name = "./llms/llama2/llama-2-7b-hf"
    max_words = 1024
    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
    with open(data_file, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    # Calculate and store weights during initialization
    new_data = []
    for i, ann in enumerate(tqdm(data, colour='white', desc='EDIT Distance Calculation')):
        # print(ann)
        chosen_input_ids, _, _ = custom_tokenize(ann['input'], ann['input'] + ann['chosen'], tokenizer, max_words)
        rejected_input_ids, _, _ = custom_tokenize(ann['input'], ann['input'] + ann['rejected'], tokenizer, max_words)

        chosen_weights, chosen_operations = assign_weights(chosen_input_ids, rejected_input_ids, 'chosen')
        rejected_weights, rejected_operations = assign_weights(rejected_input_ids, chosen_input_ids, 'rejected')

        ann['chosen_weights'] = chosen_weights.tolist() # no use, we'll reassign the weight in the sft phase
        ann['rejected_weights'] = rejected_weights.tolist() # no use
        ann['chosen_operations'] = chosen_operations # note, operation here not truncate or remove the "insert", here is a completed operation
        ann['rejected_operations'] = rejected_operations # note, operation here not truncate or remove the "insert", here is a completed operation
        new_data.append(ann)
    # with open(data_file.replace('preference', 'preference_precal'), 'w', encoding='utf-8') as f:
    #     json.dump(new_data, f, ensure_ascii=False, indent=4)
    with open(data_file.replace('preference_with_answer.json', 'preference_with_answer_precal.pkl'), 'wb') as file:
        pickle.dump(new_data, file)

    # with open(data_file.replace('preference.json', 'preference_precal.pkl'), 'rb') as file:
    #     print(pickle.load(file))

if __name__ == '__main__':
    main()
    # data_file = './dataset/bbh/bbh_all_data/all_task_train_preference.json'
    # with open(data_file.replace('preference.json', 'preference_precal.pkl'), 'rb') as file:
    #     data = pickle.load(file)
    # print(data)
