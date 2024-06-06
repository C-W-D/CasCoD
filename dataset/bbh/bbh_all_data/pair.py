# pair dual cot dataset for edit
import json
import os
import copy
data_pos_pos_file = './all_task_train_right_answer.json'
data_neg_neg_file = './all_task_train_wrong_answer.json'
data_pos_neg_file = './all_task_train_right_answer_then_wrong.json'
data_neg_pos_file = './all_task_train_wrong_answer_hint_right.json'
PROMPT_DICT = {
    "with_task_description": (
        "Task Description:\n{task_description}\nQ:{instruction}\n\nA:"
    ),
}
with open(data_pos_pos_file, "r", encoding="utf-8") as json_file:
    data_pos_pos = json.load(json_file)
with open(data_neg_neg_file, "r", encoding="utf-8") as json_file:
    data_neg_neg = json.load(json_file)
with open(data_pos_neg_file, "r", encoding="utf-8") as json_file:
    data_pos_neg = json.load(json_file)
with open(data_neg_pos_file, "r", encoding="utf-8") as json_file:
    data_neg_pos = json.load(json_file)
# double pos neg data structure
edit_data_list = []
for data in data_pos_pos:
    data_ = None
    for data_i in data_pos_neg:
        if data['instruction'] == data_i['instruction']:
            data_ = copy.deepcopy(data_i)
            break
    if data_ is None:
        continue
    else:
        new_data = {
            'input': PROMPT_DICT['with_task_description'].format_map(data),
            'chosen': data['response'],
            'rejected': data_['response']
        }
        edit_data_list.append(new_data)

for data in data_neg_pos:
    data_ = None
    for data_i in data_neg_neg:
        if data['instruction'] == data_i['instruction']:
            data_ = copy.deepcopy(data_i)
            break
    if data_ is None:
        continue
    else:
        new_data = {
            'input': PROMPT_DICT['with_task_description'].format_map(data),
            'chosen': data['response'],
            'rejected': data_['response']
        }
        edit_data_list.append(new_data)


with open('./all_task_train_preference_with_answer.json', 'w', encoding='utf-8') as f:
    json.dump(edit_data_list, f, ensure_ascii=False, indent=4)
