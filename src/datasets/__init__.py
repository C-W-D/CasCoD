# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import sys
sys.path.append('../..')

### bbh task
from src.datasets.bbh_dataset import EvalDataset as get_bbh_eval_dataset
from src.datasets.bbh_dataset import InstructionDataset as get_bbh_dataset
from src.datasets.bbh_dataset import LLMSTDataset as get_bbh_llmst_dataset
from src.datasets.bbh_dataset import LLMSTDataset as get_bbh_llmstepst_dataset
from src.datasets.bbh_dataset import LLMWeightSTDataset as get_bbh_llmweightst_dataset
from src.datasets.bbh_dataset import LLMSCOTTDataset as get_bbh_llmscott_dataset
from src.datasets.bbh_dataset import LLMMTDataset as get_bbh_llmmt_dataset
from src.datasets.bbh_dataset import LLMMTReDataset as get_bbh_llmmtre_dataset
from src.datasets.bbh_dataset import LLMMTRaDataset as get_bbh_llmmtra_dataset
from src.datasets.bbh_dataset import LLMMTCoTDataset as get_bbh_llmmtcot_dataset
from src.datasets.bbh_dataset import LLMCMTDataset as get_bbh_llmcmt_dataset
from src.datasets.bbh_dataset import KRSLDataset as get_krsl_dataset

## bb task
from src.datasets.bb_dataset import EvalDataset as get_bb_eval_dataset

## agieval task
from src.datasets.agieval_dataset import EvalDataset as get_agieval_eval_dataset

## arc task
from src.datasets.arc_dataset import EvalDataset as get_arc_eval_dataset

### utils
from src.datasets.utils import *