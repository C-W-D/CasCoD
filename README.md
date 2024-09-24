## CasCoD
Official repo of our paper: Improve Student's Reasoning Generalizability through Cascading Decomposed CoTs Distillation

## Note
This repository, developed based on [llama-recipes](https://github.com/Meta-Llama/llama-recipes), is shared with our annother CoTs distillation work, [EDIT](https://github.com/C-W-D/EDIT).

## News
Accepted as the main conference long paper in EMNLP2024. [2024/09/20]

The code, data and prompts is available now. [2024/06/06]

## TODO
More detailed guidence is in progress.

## Train CasCoD
For distilling SLMs with CasCoD, you need run the scripts:
```
cd shell
./run_distilled_cot.sh
```
The parameter dataset includes training data and distillation method settings, ranging from `[bbh_llmcmt_dataset, bbh_llmmtcot_dataset, bbh_llmmtra_dataset, bbh_llmmtre_dataset, bbh_llmscott_dataset, bbh_krsl_dataset, bbh_llmst_dataset, bbh_llmstepst_dataset, bbh_dataset]`.
The meaning are as follows：
| Alias      | Description                                 |
|--------------|---------------------------------------------|
| llmcmt       | our proposed method CasCoD                 |
| llmmtcot     | MT-CoT                                      |
| llmmtra      | MT-Ra                                       |
| llmmtre      | MT-Re / Step-by-step                        |
| llmscott     | SCOTT                                       |
| krsl         | KRSL, which is the second step in EDIT      |
| llmst        | Std-CoT                                     |
| bbh_dataset  | Answer SFT                                  |


## Train EDIT
For distilling SLMs with EDIT, you need to first run the following command to determine edit operations through the backtracking process of the minimum edit distance algorithm based on dynamic programming, generating the `.pkl` file offline.
```
python edit_dis_precal.py
```
Then, execute the first step of EDIT, a supervised fine-tuning of a base CoT model:
```
cd shell
./run_edit_step1.sh
```
Finally, execute the second step of EDIT (key reasoning step learning):
```
./run_edit_step2.sh
```
You can also visit `./dataset/bbh/cot-ahp`, `cot-ccp`, `cot-prompts` to see the prompts we used. AHP and CCP are additionally proposed and used in the EDIT work.

## Eval
For evaluating the tuned models, you can run the following command:
```
./eval_distilled_cot.sh
```
You need to change the parameter `saved_model_dir` to the path of the fine-tuned model checkpoints, `train_dataset` to the training data and distillation method settings of the fine-tuned model, and `test_dataset` to the dataset to be evaluated (ranging from `[bbh_eval_dataset, bb_eval_dataset, agieval_eval_dataset, arcc_eval_dataset, arce_eval_dataset]`).

## Citation

If you find our work helpful in your research, please star and consider citing:

```bibtex
@article{dai2024improve,
  title={Improve Student's Reasoning Generalizability through Cascading Decomposed CoTs Distillation},
  author={Dai, Chengwei and Li, Kun and Zhou, Wei and Hu, Songlin},
  journal={arXiv preprint arXiv:2405.19842},
  year={2024}
}

