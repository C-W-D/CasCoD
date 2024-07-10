from ChatGPTAPI import *
import json
from tqdm import tqdm

# repair the uncorrect structure of the response

import os

data_dir = "../dataset/bbh/bbh_data"  # 数据目录
call_data_file_list = []  # 存储文件路径的列表

# 遍历数据目录
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file in ['train_data.json', 'test_data.json']:
            file_path = os.path.join(root, file)  # 获取文件的绝对路径
            call_data_file_list.append(file_path)

# # 输出call_data_file_list
# for file_path in call_data_file_list:
#     print(file_path)

call_result_file = ''


for cdf in call_data_file_list:
    print(cdf)
    call_result_file = ''
    for i in cdf.split('/')[:-1]:
        call_result_file = os.path.join(call_result_file, i)
    call_result_file = os.path.join(call_result_file, 'call_result_') + cdf.split('/')[-1]
    print(call_result_file)
    api_instance = ChatGPTAPI(call_result_file)

    with open(cdf, "r", encoding="utf-8") as json_file:
        datas = json.load(json_file)

    # datas = datas[:1]
    task_name = cdf.split('/')[-2]
    prompt_example_file = f'../dataset/bbh/cot-prompts/{task_name}.txt'
    with open(prompt_example_file, 'r') as file:
        prompt = file.read()
    print("prompt:", prompt)

    new_datas = []
    for data in tqdm(datas, colour='blue'):

        base_num = 1
        base_t = 0.1
        flag = True
        while flag:
            base_num += 1
            base_num = min(10, base_num)
            base_t += 0.1
            base_t = min(1.5, base_t)
            api_instance.set_msgnum(base_num)
            api_instance.set_temperature(base_t)
            responses = api_instance.access_api(prompt.replace('{QUESTION}', data['input']))['choices']
            for r in responses:
                if 'Therefore, the answer is' in r['message']['content']:
                    flag = False
                    response = r['message']['content']
                    break

        new_datas.append({
            'instruction': data['input'],
            'input': '',
            'output': data['target'],
            'prompt': prompt.replace('{QUESTION}', data['input']),
            'response': response,
            'task_description': data['task_description']
        })

    with open(cdf.replace('_data.json', '_data_wigpt.json'), 'w', encoding='utf-8') as f:
        json.dump(new_datas, f, ensure_ascii=False, indent=4)

    print("save file:", cdf.replace('_data.json', '_data_wigpt.json'))