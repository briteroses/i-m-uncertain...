import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from datasets import Dataset

from copy import deepcopy

from data.contrast import UncertaintyContrastDataset, IDK_DUMMY_LABEL
from data.prompts import negAndPosLabels
from data.registry import SAVE_PREFIX, get_label_name_for_dataset

ESTIMATED_TRAINING_CUTOFFS = {
    'deberta': '2/10/2021',
    'gpt-j': '8/31/2021',
    'gpt2-large': '8/20/2019',
    'T0pp': '10/12/2021',
    'unifiedqa': '11/13/2020',
}

PROMPTS_FOR_TEMPORAL = [
    (
        "Consider the following statement: ''' {} '''\nBetween true and false, this statement is ",
        lambda text: text.capitalize(),
    ),
    (
        "Consider the following statement: ''' {} '''\nIs this statement true or false?",
        lambda text: text.capitalize(),
    ),
    (
        "Is it true or false that {}?",
        lambda text: text.rstrip('.'),
    ),
]

class TemporalDataset(UncertaintyContrastDataset):
    def __init__(self, *args, **kwargs):
        # can just use exact same init as contrast dataset
        super().__init__(*args, **kwargs)

    # inherited method is a misnomer here: the prompting happens in the main method using this dataset class
    def get_prompts_at_index(self, index):
        question = self.raw_dataset[int(index)]['question']

        neg_prompt = (question, "true")
        pos_prompt = (question, "false")
        idk_prompt = (question, self.idk_word)

        return neg_prompt, pos_prompt, idk_prompt


def temporal_dataloader(dataset_name, temporal_split, tokenizer, prompt_idx,
                            model_name="deberta", use_decoder=False,
                            device="cuda", pin_memory=True, num_workers=1,
                            idk_word="uncertain"):
    assert temporal_split in ("raw", "masked"), "need to specify between raw or masked dataset variant"

    path_to_temporal_dataset = SAVE_PREFIX + 'temporal_dataset.csv'
    df = pd.read_csv(path_to_temporal_dataset)
    texts = df['text'].tolist()
    labels = df['label'].astype(int).values
    timestamps = pd.to_datetime(df['timestamp'], format='%m/%d/%Y').values

    template, text_preprocessor, truth_words = PROMPTS_FOR_TEMPORAL[prompt_idx]
    questions = map(lambda text: template.format(text_preprocessor(text)), texts)
    
    if temporal_split == 'raw':
        raw_dataset = Dataset.from_dict({
            'question': questions,
            'label': labels
        })

        ds = TemporalDataset(raw_dataset, tokenizer, None, idk_word=idk_word,
                                model_name=model_name, dataset_name=dataset_name, use_decoder=use_decoder, 
                                device=device)
        
        loader = DataLoader(ds, batch_size=16, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)

        return loader

    elif temporal_split == 'masked':
        training_cutoff = pd.to_datetime(ESTIMATED_TRAINING_CUTOFFS[model_name], format='%m/%d/%Y')
        temporal_mask = timestamps > training_cutoff
        temporal_masked_labels = np.where(temporal_mask, IDK_DUMMY_LABEL, labels)

        raw_temporal_masked_dataset = Dataset.from_dict({
            'question': questions,
            'label': temporal_masked_labels
        })
        temporal_masked_ds = TemporalDataset(raw_temporal_masked_dataset, tokenizer, None, idk_word=idk_word,
                                            model_name=model_name, dataset_name=dataset_name, use_decoder=use_decoder, 
                                            device=device)
        temporal_masked_loader = DataLoader(temporal_masked_ds, batch_size=16, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)

        return temporal_masked_loader