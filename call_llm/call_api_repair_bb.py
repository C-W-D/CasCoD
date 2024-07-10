from ChatGPTAPI import *
import json
from tqdm import tqdm
import copy
# repair the uncorrect structure of the response

import os

data_dir = "../dataset/bb/merged_data/bb_sub_task_random100test.json"  # 数据目录


call_result_file = ''
for i in data_dir.split('/')[:-1]:
    call_result_file = os.path.join(call_result_file, i)
call_result_file = os.path.join(call_result_file, 'call_result_') + data_dir.split('/')[-1]
print(call_result_file)
api_instance = ChatGPTAPI(call_result_file)

with open(data_dir, "r", encoding="utf-8") as json_file:
    datas = json.load(json_file)

try:
    with open(call_result_file, "r", encoding="utf-8") as json_file:
        history = json.load(json_file)
except Exception as e:
    print(e)
    history = []

history_data = {}
for h_data in history:
    if 'status' in h_data.keys():
        if h_data['status'] == 'failed':
            datas
    if 'choices' not in h_data.keys():
        continue
    history_data[h_data['user_prompt']] = h_data['choices'][0]['message']['content']

new_datas = []
for data in tqdm(datas, colour='blue'):
    prompt = data['task_description'] + ' Your response needs to give the thought first, and then should conclude with the format "Therefore, the answer is" in the end.\n\n'
    prompt += 'Q: ' + data['instruction'] + '\n\nA: Let\'s think step by step.'
    
    if prompt in history_data.keys():
        new_data = copy.deepcopy(data)
        new_data['prompt'] = prompt
        new_data['response'] = history_data[prompt]
        new_datas.append(new_data)
        print('exist')
        continue

    base_num = 0
    base_t = 0.1
    flag = True
    while flag:
        base_num += 1
        base_num = min(10, base_num)
        base_t += 0.1
        base_t = min(1.5, base_t)
        api_instance.set_msgnum(base_num)
        api_instance.set_temperature(base_t)
        responses = api_instance.access_api(prompt)
        if 'choices' not in responses.keys():
            break
        responses = responses['choices']
        for r in responses:
            if 'Therefore, the answer is' in r['message']['content']:
                flag = False
                response = r['message']['content']
                break

    if flag:
        print('skip')
        continue
    new_data = copy.deepcopy(data)
    new_data['prompt'] = prompt
    new_data['response'] = response
    new_datas.append(new_data)

with open(data_dir, 'w', encoding='utf-8') as f:
    json.dump(new_datas, f, ensure_ascii=False, indent=4)
