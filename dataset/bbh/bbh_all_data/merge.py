# merge json file
import json
def merge(f1, f2, of):
    # read json data from f1 and f2, then save in of
    with open(f1, 'r', encoding='utf-8') as file1:
        data1 = json.load(file1)
    with open(f2, 'r', encoding='utf-8') as file2:
        data2 = json.load(file2)
    
    merged_data = data1 + data2
    with open(of, 'w', encoding='utf-8') as output_file:
        json.dump(merged_data, output_file, ensure_ascii=False, indent=4)

original_right_file = './all_task_train_right_answer.json'
rectified_file = './all_task_train_wrong_answer_hint_right.json'
output_file = './all_task_train_right_wronghint_answer.json'
merge(original_right_file, rectified_file, output_file)