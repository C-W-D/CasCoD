from ChatGPTAPI import *
import json
from tqdm import tqdm
import copy
# repair the uncorrect structure of the response

import os

data_dir = "../dataset/bbh/bbh_all_data/all_task_test.json"  # 数据目录
save_file = "../dataset/bbh/bbh_all_data/all_task_test_0shot.json"
api_key = ''

call_result_file = ''
for i in data_dir.split('/')[:-1]:
    call_result_file = os.path.join(call_result_file, i)
call_result_file = os.path.join(call_result_file, 'call_result_') + data_dir.split('/')[-1]
print(call_result_file)
api_instance = ChatGPTAPI(call_result_file)
api_instance.set_model_engine('gpt-3.5-turbo-16k-0613')
api_instance.set_api_key(api_key)

with open(data_dir, "r", encoding="utf-8") as json_file:
    datas = json.load(json_file)

new_datas = []
for data in tqdm(datas, colour='blue'):
    prompt = 'Answer the following question.' + ' Your response needs to give the thought first, and then should conclude with the format "Therefore, the answer is" in the end.\n\n'
    prompt += 'Q: ' + data['instruction'] + '\n\nA: Let\'s think step by step.'
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
        responses = api_instance.access_api(prompt)['choices']
        for r in responses:
            if 'Therefore, the answer is' in r['message']['content']:
                flag = False
                response = r['message']['content']
                break

    new_data = copy.deepcopy(data)
    new_data['prompt'] = prompt
    new_data['response'] = response
    new_datas.append(new_data)

with open(save_file, 'w', encoding='utf-8') as f:
    json.dump(new_datas, f, ensure_ascii=False, indent=4)
