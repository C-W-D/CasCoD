from dataclasses import dataclass

@dataclass
class bbh_eval_dataset:
    dataset: str = "bbh_eval_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "null"
    test_data_path: str = "./dataset/bbh/bbh_all_data/all_task_test.json"
    max_words: int = 150

@dataclass
class bbhtrain_eval_dataset:
    dataset: str = "bbh_eval_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "null"
    test_data_path: str = "./dataset/bbh/bbh_all_data/all_task_train_right_answer.json"
    max_words: int = 150

@dataclass
class bbh_dataset:
    dataset: str = "bbh_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "./dataset/bbh/bbh_all_data/all_task_train_right_answer.json"
    test_data_path: str = "./dataset/bbh/bbh_all_data/all_task_test.json"
    max_words: int = 150


@dataclass
class bbh_llmst_dataset:
    dataset: str = "bbh_llmst_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "./dataset/bbh/bbh_all_data/all_task_train_right_answer.json"
    test_data_path: str = "./dataset/bbh/bbh_all_data/all_task_test.json"
    max_words: int = 150

@dataclass
class bbh_llmweightst_dataset:
    dataset: str = "bbh_llmweightst_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "./dataset/bbh/bbh_all_data/all_task_train_right_answer.json"
    test_data_path: str = "./dataset/bbh/bbh_all_data/all_task_test.json"
    max_words: int = 150
    weight_type: str = 'linear_decrease'
    step_type: str='sentence_split'


@dataclass
class bbh_llmmt_dataset:
    dataset: str = "bbh_llmmt_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "./dataset/bbh/bbh_all_data/all_task_train_right_answer.json"
    test_data_path: str = "./dataset/bbh/bbh_all_data/all_task_test.json"
    max_words: int = 150

@dataclass
class bbh_llmmtre_dataset:
    dataset: str = "bbh_llmmtre_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "./dataset/bbh/bbh_all_data/all_task_train_right_answer.json"
    test_data_path: str = "./dataset/bbh/bbh_all_data/all_task_test.json"
    max_words: int = 150

@dataclass
class bbh_llmmtra_dataset:
    dataset: str = "bbh_llmmtra_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "./dataset/bbh/bbh_all_data/all_task_train_right_answer.json"
    test_data_path: str = "./dataset/bbh/bbh_all_data/all_task_test.json"
    max_words: int = 150

@dataclass
class bbh_llmmtcot_dataset:
    dataset: str = "bbh_llmmtcot_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "./dataset/bbh/bbh_all_data/all_task_train_right_answer.json"
    test_data_path: str = "./dataset/bbh/bbh_all_data/all_task_test.json"
    max_words: int = 150

@dataclass
class bbh_llmcmt_dataset:
    dataset: str = "bbh_llmcmt_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "./dataset/bbh/bbh_all_data/all_task_train_right_answer.json"
    test_data_path: str = "./dataset/bbh/bbh_all_data/all_task_test.json"
    max_words: int = 150


@dataclass
class bbh_llmstepst_dataset:
    # for one step, refet to llmst
    dataset: str = "bbh_llmstepst_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "./dataset/bbh/bbh_all_data/all_task_train_right_answer.json" # default for two step
    test_data_path: str = "./dataset/bbh/bbh_all_data/all_task_test.json"
    max_words: int = 150
    n_step: int = 2 

@dataclass
class bbh_llmscott_dataset:
    dataset: str = "bbh_llmscott_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "./dataset/bbh/bbh_all_data/all_task_train_final.json"
    test_data_path: str = "./dataset/bbh/bbh_all_data/all_task_test.json"
    max_words: int = 150


@dataclass
class bbh_krsl_dataset:
    dataset: str = "bbh_krsl_dataset"
    train_split: str = "train"
    test_split: str = "val"
    krsl_pre_dataset: str = "none"
    krsl_train_data_path: str = "./dataset/bbh/bbh_all_data/all_task_train_right_preference.json"
    krsl_weight_path: str = "./dataset/bbh/bbh_all_data/all_task_train_right_preference_precal.pkl"
    test_data_path: str = "./dataset/bbh/bbh_all_data/all_task_test.json"
    max_words: int = 150
    krsl_alpha: float = 1.0
    krsl_beta: float = -0.5
    krsl_gamma: float = 0.0
    rouge2_below: float = 1.01

@dataclass
class bb_eval_dataset:
    dataset: str = "bb_eval_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "null"
    test_data_path: str = "./dataset/bb/merged_data/bb_sub_task_random100test.json"
    max_words: int = 150

@dataclass
class agieval_eval_dataset:
    dataset: str = "agieval_eval_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "null"
    test_data_path: str = "./dataset/agieval/merged_data/all_task_test.json"
    max_words: int = 150

@dataclass
class arce_eval_dataset:
    dataset: str = "arce_eval_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "null"
    test_data_path: str = "./dataset/arc-e/merged_data/arc_easy_test.json"
    max_words: int = 150

@dataclass
class arcc_eval_dataset:
    dataset: str = "arce_eval_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "null"
    test_data_path: str = "./dataset/arc-c/merged_data/arc_challenge_test.json"
    max_words: int = 150