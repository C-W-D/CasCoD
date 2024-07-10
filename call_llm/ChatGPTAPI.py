
import openai
# openai.proxy = ''
openai.api_base = ""
# openai.api_key = ""
import os
import sys
import random
import time
import json
import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y%m%d%H%M%S")
print(formatted_time)

class ChatGPTSetting:
    # chatgpt and instruction settings
    # model_engine = "gpt-3.5-turbo-0613" # teacher
    # gpt-3.5-turbo-16k-0613
    # gpt-3.5-turbo-0613
    model_engine = "gpt-4-1106-preview" # teacher
    # model_engine = "gpt-3.5-turbo-0613" # evaluator
    temperature = 0.2
    msg_num = 1
    max_tokens = 4096
    stop = None


class ChatGPTAPI:
    def __init__(self, save_call_info_file='./call_result/example.json', role=None) -> None:
        self.save_call_info_file = save_call_info_file
        self.model_engine = ChatGPTSetting.model_engine
        self.temperature = ChatGPTSetting.temperature
        self.msg_num = ChatGPTSetting.msg_num
        self.max_tokens = ChatGPTSetting.max_tokens
        self.stop = ChatGPTSetting.stop
        # if role != 'teacher':
        self.model_engine = ChatGPTSetting.model_engine
        # else:
        #     self.model_engine = "gpt-3.5-turbo-0613"

    def set_temperature(self, temperature):
        self.temperature = temperature
    
    def set_msgnum(self, msgnum):
        self.msg_num = msgnum
        
    def set_maxtokens(self, maxtokens):
        self.max_tokens = maxtokens

    def set_model_engine(self, model_engine):
        self.model_engine = model_engine

    def set_api_key(self, key):
        openai.api_key = key

    def access_api(self, prompt, system_content=None):
        """
        输入prompt
        返回消息列表msg_list: len of list is equal to msg_num
        以及完整的completions
        """
        if system_content is not None:
            messages = [{
                        'role': 'system',
                        'content': system_content,
                    },
                    {
                        'role': 'user', 
                        'content': prompt
                }
            ]
        else:
            messages = [
                    {
                        'role': 'user', 
                        'content': prompt
                }
            ]
        max_retry = 1e6 + 5 # connect retry number
        # max_retry = 10
        cnt = 0
        flag = True
        while flag:
            try:
                completions = openai.ChatCompletion.create(
                    model=self.model_engine,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    n=self.msg_num,
                    stop=self.stop,
                    temperature=self.temperature
                )
                flag = False
                # print("access API successfully")
            except Exception as e:
                cnt = cnt + 1
                if cnt > max_retry:
                    break
                print("connection error %d times...reconnecting..." % cnt)
                print("exception: ", e)
                if 'repetitive patterns in your prompt' in str(e):
                    print(messages)
                    messages[-1]['content'] = self.find_and_slice_from_right(messages[-1]['content']) + '. \n\nYour Answer:'
                sleep_time = 5 + random.randint(0, 10)
                print("sleep %d seconds...try access again..." % sleep_time)
                time.sleep(sleep_time)
        if cnt > max_retry:
            print("connection times exceed max_retry!")
            completions = {
                'status': 'failed'
            }

        completions['user_prompt'] = prompt

        # append the completions to the save_call_info_file here
        try:
            with open(self.save_call_info_file, 'r') as file:
                existing_data = json.load(file)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            existing_data = []  # Initialize with an empty list if the file is empty or doesn't exist

        existing_data.append(completions)
 
        # Write the updated data back to the JSON file
        with open(self.save_call_info_file, 'w', encoding='utf-8') as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)
        # write above
        return completions
    def find_and_slice_from_right(self, input_str):
        # 从右往左找到第一个 ' . ' 的位置
        dot_index = input_str.rfind('.')
        
        # 如果找到了 ' . '
        if dot_index != -1:
            # 截取字符串从开头到 ' . ' 的位置
            result_str = input_str[:dot_index]
            return result_str
        else:
            # 如果没找到 ' . '，返回原始字符串
            return input_str
    
if __name__ == "__main__":
    api_key = ""
    system_content = "You are a helpful and precise assistant for following the given instruction."
    prompt=""""Answer the following multiple-choice questions. Let's think step by step. If there is no correct option, give the option that is closest to the correct answer. Your response should conclude with the format "Therefore, the answer is <answer content> <answer choice>." The questions are as follows:\nQ: What is the most famous constellation out of earth?\nAnswer Choices:\n(a) one moon\n(b) milky way\n(c) god's creation\n(d) stars\n(e) universe\n\n"""
    a=ChatGPTAPI(api_key)
    a.set_api_key(api_key)
    b=a.access_api(prompt, system_content)
    print(b)



