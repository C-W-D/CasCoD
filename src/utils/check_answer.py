# toy code
import re
def extract_answers_for_model(model_output):
    if model_output is None:
        return 'none', 'none'
    model_output = model_output.strip()
    model_output = model_output.rstrip('.')
    model_output = model_output.lower()
    model_output = model_output.replace(u'）', ')').replace(u'（', '(')

    if len(model_output) == 1:
        md_choice = f'({model_output})'
        md_content = 'none'
    else:
        pattern = r'\([a-z]\)'
        match = re.search(pattern, model_output)
        if match:
            md_choice = match.group(0)
            sp_list = model_output.split(')')
            if len(sp_list) == 1:
                md_content = 'none'
            else:
                md_content = sp_list[1]
                md_content = md_content.lstrip('.')
                md_content = md_content.rstrip('.')
                md_content = md_content.strip()
        else:
            md_choice = 'none'
            md_content = model_output.lstrip('.')
            md_content = md_content.rstrip('.')
            md_content = md_content.strip()

    return md_choice, md_content


def extract_answers_for_gt(original_output):
    original_output = original_output.strip()
    original_output = original_output.rstrip('.')
    original_output = original_output.lower()
    original_output = original_output.replace(u'）', ')').replace(u'（', '(')

    pattern = r'\([a-z]\)'
    match = re.search(pattern, original_output)
    if match:
        gt_choice = match.group(0)
        sp_list = original_output.split(')')
        if len(sp_list) == 1:
            gt_content = 'none'
        else:
            gt_content = sp_list[1]
            gt_content = gt_content.lstrip('.')
            gt_content = gt_content.rstrip('.')
            gt_content = gt_content.strip()
    else:
        gt_choice = 'none'
        gt_content = original_output.lstrip('.')
        gt_content = gt_content.rstrip('.')
        gt_content = gt_content.strip()

    return gt_choice, gt_content

original_output = '(A)'
model_output = ''

gt_choice, gt_content = extract_answers_for_gt(original_output)
md_choice, md_content = extract_answers_for_model(model_output)

# note: gt choice must be parthed with '()', like '(A)'
# but md choice are not required, likr 'A'

if md_choice and md_choice != 'none':
    if md_choice in gt_choice:
        # check choice
        print('true')
    else:
        print('false')
else:
    # check content
    if gt_content == md_content:
        print('true')
    else:
        print('false')